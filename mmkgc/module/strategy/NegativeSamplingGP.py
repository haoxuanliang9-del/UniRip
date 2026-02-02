from .Strategy import Strategy


class NegativeSamplingGP(Strategy):
    def __init__(
        self,
        model=None,
        loss=None, 
        batch_size=256, 
        regul_rate=0.0, 
        l3_regul_rate=0.0,
        struct_cons_rate=0.0
    ):
        super(NegativeSamplingGP, self).__init__()
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        self.regul_rate = regul_rate
        self.l3_regul_rate = l3_regul_rate
        self.struct_cons_rate = struct_cons_rate

    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size]
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        return positive_score

    def _get_negative_score(self, score):
        negative_score = score[self.batch_size:]
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        return negative_score

    def forward(self, data, fast_return=False):
        score = self.model(data)
        p_score = self._get_positive_score(score)
        if fast_return:
            return p_score
        n_score = self._get_negative_score(score)
        # print('P_score:',p_score.mean().item(),'n_score:',n_score.mean().item())
        loss_res = self.loss(p_score, n_score)
        if self.regul_rate != 0:
            loss_res += self.regul_rate * self.model.regularization(data)
        if self.l3_regul_rate != 0:
            loss_res += self.l3_regul_rate * self.model.l3_regularization()
        if self.struct_cons_rate != 0 and hasattr(self.model, "get_struct_consistency_loss"):
            loss_res += self.struct_cons_rate * self.model.get_struct_consistency_loss()
        return loss_res
