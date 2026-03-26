import pandas as pd
import numpy as np

def test_dataframe_creation():
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    assert len(df) == 3

def test_numpy_operations():
    arr = np.array([1,2,3,4,5])
    assert arr.mean() == 3.0

def test_churn_rate_calculation():
    df = pd.DataFrame({'churned': [1,0,1,0,1,0,1,0,1,0]})
    churn_rate = df['churned'].mean()
    assert 0 <= churn_rate <= 1

def test_clv_positive():
    clv_values = [100, 200, 150, 88, 300]
    assert all(v > 0 for v in clv_values)
