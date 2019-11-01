
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
            log_probs = self.log_probs + [log_probs],
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
    hyps = [Hypothesis(
                        tokens=[vocab.word2id(data.START_DECODING)],
                        log_probs=[0.0],                        
                        state=dec_in_state,                     
                        attn_dists=[],
                        p_gens=[],                              
                        coverage=np.zeros([batch.enc_batch.shape[1]]) # zero vector of length attention_length     
                        ) for _ in range(FLAGS.beam_size)]
    results = []

    steps = 0
    while steps < max_dec_steps and len(results) < beam_size:
        latest_tokens = [h.latest_token for h in hyps]
        latest_tokens = [ t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens ]
        states = [ h.state for h in hyps ]
        (topk_ids, topk_log_probs, new_states, attn_dists) = model.decode_onestep(
                batch=batch,
                latest_tokens=latest_tokens,
                enc_states=enc_states
            )
        all_hyps=[]
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        for i in range(num_orig_hyps):
            h, new_state, attn_dist = hyps[i], new_states[i], attn_dists[i]
            for j in range(beam_size * 2):
                new_hyp = h.extend(
                        token=topk_ids[i,j],
                        log_prob=topk_log_probs[i,j],
                        state=new_state,
                        attn_dist=attn_dist
                    )
                all_hyps.append(new_hyp)
        hyps=[]
        for h in sort_hyps(all_hyps):
            if h.latest_token == vocab.word2id(data.STOP_DECODING):
                if steps >= FLAGS.min_dec_steps:
                    results.append(h)
            else:
                hyps.append(h)
            if len(hyps) == beam_size or len(results) == beam_size:
                break
        steps += 1

    if len(results) == 0:
        results = hyps
    hyps_sorted = sort_hyps(results)
    return hyps_sorted[0]

def sort_hyps(hyps):
    return sorted(hyps, key=lambda h:h.avg_log_prob, reverse=True)
    
