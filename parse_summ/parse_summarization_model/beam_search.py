from mxnet import nd


class Hypothesis():

    def __init__(self, tokens, log_probs, state, attn_dists):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
    
    def extend(self, token, log_prob, state, attn_dist):
        return Hypothesis(
            tokens = self.tokens + [token],
            state = state,
            attn_dists = self.attn_dists + [attn_dist],
        )
    
    @property
    def latest_token(self):
        return self.tokens[-1]
    
    @property
    def log_prob(self):
        return sum(self.log_probs)
    
    @property
    def avg_log_prob(self):
        return self.log_prob / len(self.tokens)


def run_beam_search(model, vocab, batch):
    enc_states, dec_in_state = model.run_encoder(batch)

    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],                     log_probs=[0.0],                        
                        state=dec_in_state,                     
                        attn_dists=[],
                        p_gens=[],                              
                        coverage=np.zeros([batch.enc_batch.shape[1]]) # zero vector of length attention_length     
                        ) for _ in range(FLAGS.beam_size)]
    results = []

    steps = 0
    while steps < max_dec_steps and len(results) < beam_size:
        latest_tokens = [h.latest_token for h in hyps]

    
