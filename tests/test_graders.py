from server.grader import LedgerGrader, SubscriptionGrader, CashFlowGrader

ledger_grader = LedgerGrader()
subscription_grader = SubscriptionGrader()
cash_flow_grader = CashFlowGrader()

class TestLedgerGrader:
    def test_perfect(self):
        assert ledger_grader(50, 50) == 0.999

    def test_half(self):
        assert ledger_grader(25, 50) == 0.5

    def test_zero(self):
        assert ledger_grader(0, 50) == 0.001

    def test_zero_baseline(self):
        assert ledger_grader(10, 0) == 0.001

class TestSubscriptionGrader:
    def test_perfect(self):
        assert subscription_grader(2, 2) == 0.999
        
    def test_half(self):
        assert subscription_grader(1, 2) == 0.5
        
    def test_zero(self):
        assert subscription_grader(0, 2) == 0.001
        
    def test_zero_baseline(self):
        assert subscription_grader(1, 0) == 0.999

class TestCashFlowGrader:
    def test_perfect(self):
        assert cash_flow_grader(500, 500) == 0.999
        
    def test_half(self):
        assert cash_flow_grader(250, 500) == 0.5
        
    def test_zero(self):
        assert cash_flow_grader(0, 500) == 0.001
        
    def test_zero_baseline(self):
        assert cash_flow_grader(100, 0) == 0.5
