from server import grader

class TestLedgerGrader:
    def test_perfect(self):
        assert grader.grade_ledger(50, 50) == 0.999

    def test_half(self):
        assert grader.grade_ledger(25, 50) == 0.5

    def test_zero(self):
        assert grader.grade_ledger(0, 50) == 0.001

    def test_zero_baseline(self):
        assert grader.grade_ledger(10, 0) == 0.001

class TestSubscriptionGrader:
    def test_perfect(self):
        assert grader.grade_subscription(2, 2) == 0.999
        
    def test_half(self):
        assert grader.grade_subscription(1, 2) == 0.5
        
    def test_zero(self):
        assert grader.grade_subscription(0, 2) == 0.001
        
    def test_zero_baseline(self):
        assert grader.grade_subscription(1, 0) == 0.999

class TestCashFlowGrader:
    def test_perfect(self):
        assert grader.grade_cash_flow(500, 500) == 0.999
        
    def test_half(self):
        assert grader.grade_cash_flow(250, 500) == 0.5
        
    def test_zero(self):
        assert grader.grade_cash_flow(0, 500) == 0.001
        
    def test_zero_baseline(self):
        assert grader.grade_cash_flow(100, 0) == 0.5
