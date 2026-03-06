use ndarray::Array2;

use crate::{MAX_SEQ_LEN, utils::softmax};

use super::LLM;

#[derive(Clone)]
/// 推理会话的快照（用于保存/恢复 KV cache 与已处理 token 数）。
///
/// 该类型需要是公开的，否则 `InferenceSession::snapshot/restore` 的 public API 会泄漏私有类型。
pub struct SessionSnapshot {
    pub(super) processed_tokens: usize,
    pub(super) kv_caches: Vec<Option<(Array2<f32>, Array2<f32>)>>,
}

pub struct InferenceSession<'a> {
    llm: &'a mut LLM,
    processed_tokens: usize,
    max_context_length: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    previous_training_mode: bool,
    kv_cache_was_enabled: bool,
}

impl<'a> InferenceSession<'a> {
    pub fn new(llm: &'a mut LLM, temperature: f32, top_p: f32, top_k: usize) -> Self {
        let previous_training_mode = llm.training;
        let kv_cache_was_enabled = llm.is_kv_cache_enabled();
        let max_context_length = llm.max_context_length;

        llm.set_training_mode(false);
        llm.enable_kv_cache();
        llm.clear_kv_cache();

        Self {
            llm,
            processed_tokens: 0,
            max_context_length,
            temperature,
            top_p,
            top_k,
            previous_training_mode,
            kv_cache_was_enabled,
        }
    }

    pub fn prime_tokens(&mut self, tokens: &[usize]) -> Option<Array2<f32>> {
        if tokens.is_empty() {
            return None;
        }

        self.llm.clear_kv_cache();
        self.processed_tokens = 0;

        let mut last_logits = None;
        for &token in tokens {
            last_logits = Some(self.advance_with_token(token));
        }

        last_logits
    }

    /// 推进一步增量推理。
    ///
    /// 教学边界说明：
    /// - 当前实现到达 `max_context_length` 后会**整段清空 KV cache 并从当前位置重新计数**；
    /// - 这不是滑动窗口推理，只是一个保持实现简单的 hard reset 策略。
    pub fn advance_with_token(&mut self, token_id: usize) -> Array2<f32> {
        if self.processed_tokens >= self.max_context_length {
            log::warn!(
                "上下文长度超过阈值({}), 自动重置 KV 缓存",
                self.max_context_length
            );
            self.llm.clear_kv_cache();
            self.processed_tokens = 0;
        }

        let logits = self.llm.inference_step(token_id, self.processed_tokens);
        self.processed_tokens += 1;
        logits
    }

    pub fn sample_next_token(&mut self, logits: &Array2<f32>) -> usize {
        let probs = softmax(logits);
        self.llm
            .sample_token_from_probs(&probs, self.temperature, self.top_p, self.top_k)
    }

    pub fn snapshot(&mut self) -> SessionSnapshot {
        SessionSnapshot {
            processed_tokens: self.processed_tokens,
            kv_caches: self.llm.capture_kv_cache(),
        }
    }

    pub fn restore(&mut self, snapshot: &SessionSnapshot) {
        self.processed_tokens = snapshot.processed_tokens;
        self.llm.restore_kv_cache(&snapshot.kv_caches);
    }

    pub fn processed_tokens(&self) -> usize {
        self.processed_tokens
    }
}

impl<'a> Drop for InferenceSession<'a> {
    fn drop(&mut self) {
        if self.previous_training_mode {
            self.llm.set_training_mode(true);
        } else {
            self.llm.set_training_mode(false);
        }

        if !self.kv_cache_was_enabled {
            self.llm.disable_kv_cache();
        }

        self.llm.clear_kv_cache();
    }
}

impl LLM {
    #[allow(dead_code)]
    pub fn predict(&mut self, text: &str) -> String {
        self.predict_with_sampling(text, 1.0, 0.9, 5)
    }

    #[allow(dead_code)]
    pub fn predict_with_sampling(
        &mut self,
        text: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        let prompt_tokens = self.tokenize(text);
        let max_new_tokens = MAX_SEQ_LEN.saturating_sub(prompt_tokens.len());
        let generated_tokens = self.generate_tokens_incremental(
            &prompt_tokens,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        );

        self.tokens_to_text(&generated_tokens)
    }

    pub fn predict_with_context(
        &mut self,
        text: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        let new_tokens = self.tokenize(text);

        let mut combined_tokens = self.context_window.clone();
        combined_tokens.extend_from_slice(&new_tokens);

        if combined_tokens.len() > self.max_context_length {
            let start_idx = combined_tokens.len() - self.max_context_length;
            combined_tokens = combined_tokens[start_idx..].to_vec();
        }

        let available = self
            .max_context_length
            .saturating_sub(combined_tokens.len());
        let max_new_tokens = available.min(20);
        let generated_tokens = self.generate_tokens_incremental(
            &combined_tokens,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        );

        self.add_to_context(&new_tokens);
        self.add_to_context(&generated_tokens);

        self.tokens_to_text(&generated_tokens)
    }

    pub fn predict_with_beam_search(
        &mut self,
        text: &str,
        beam_width: usize,
        max_length: usize,
    ) -> String {
        self.beam_search(text, beam_width, max_length)
    }
}
