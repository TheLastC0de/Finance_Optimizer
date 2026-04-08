from graders import ledger_grader, subscription_grader, cash_flow_grader

class TestLedgerGrader:
    def test_perfect(self):
        assert ledger_grader.grade(50, 50) == 0.999

    def test_half(self):
        assert ledger_grader.grade(25, 50) == 0.5

    def test_zero(self):
        assert ledger_grader.grade(0, 50) == 0.001

    def test_zero_baseline(self):
        assert ledger_grader.grade(10, 0) == 0.001

class TestSubscriptionGrader:
    def test_perfect(self):
        assert subscription_grader.grade(2, 2) == 0.999
        
    def test_half(self):
        assert subscription_grader.grade(1, 2) == 0.5
        
    def test_zero(self):
        assert subscription_grader.grade(0, 2) == 0.001
        
    def test_zero_baseline(self):
        assert subscription_grader.grade(1, 0) == 0.999

class TestCashFlowGrader:
    def test_perfect(self):
        assert cash_flow_grader.grade(500, 500) == 0.999
        
    def test_half(self):
        assert cash_flow_grader.grade(250, 500) == 0.5
        
    def test_zero(self):
        assert cash_flow_grader.grade(0, 500) == 0.001
        
    def test_zero_baseline(self):
        assert cash_flow_grader.grade(100, 0) == 0.5
