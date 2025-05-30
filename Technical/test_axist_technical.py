import unittest
import pandas as pd
import numpy as np
import sys
import os
import importlib.util

# Attempt to load axist_technical.py using importlib
module_name = "axist_technical"

# Construct path from CWD
cwd = os.getcwd() # Should be /app when script is run with 'cd /app'
path_to_module_absolute = os.path.join(cwd, "Technical", module_name + ".py")

print(f"DEBUG: CWD is {os.getcwd()}")
print(f"DEBUG: __file__ is {__file__}") # This will be relative to CWD now: Technical/test_axist_technical.py
print(f"DEBUG: Trying to load module from constructed absolute path: {path_to_module_absolute}")
print(f"DEBUG: Does file exist? {os.path.exists(path_to_module_absolute)}")

spec = None
axist_module = None
functions_loaded = False

try:
    if not os.path.exists(path_to_module_absolute):
        raise FileNotFoundError(f"File not found at constructed path: {path_to_module_absolute}")

    spec = importlib.util.spec_from_file_location(module_name, path_to_module_absolute)
    if spec and spec.loader:
        axist_module = importlib.util.module_from_spec(spec)
        if axist_module:
            # Before exec_module, make sure its own imports can be resolved if they are relative
            # or need 'Technical' package context. Add /app to sys.path if not there.
            project_root = os.path.abspath(os.path.join(os.path.dirname(path_to_module_absolute), '..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # Also add the 'Technical' directory itself to sys.path for its internal imports if any
            technical_pkg_dir = os.path.dirname(path_to_module_absolute)
            if technical_pkg_dir not in sys.path:
                 sys.path.insert(1, technical_pkg_dir) # Insert after project_root

            print(f"DEBUG: sys.path before exec_module: {sys.path}")

            sys.modules[module_name] = axist_module
            spec.loader.exec_module(axist_module)
            print(f"Module '{module_name}' loaded successfully using importlib.")
            compute_indicators = axist_module.compute_indicators
            prepare_features = axist_module.prepare_features
            generate_signal_output = axist_module.generate_signal_output
            functions_loaded = True
            if hasattr(axist_module, 'Fore'): Fore = axist_module.Fore
            if hasattr(axist_module, 'Style'): Style = axist_module.Style
            if 'Fore' not in globals(): from colorama import Fore, Style
        else:
            raise ImportError(f"Could not create module from spec for {module_name}")
    else:
        raise ImportError(f"Could not create spec or loader for {module_name} at {path_to_module_absolute}")
except Exception as e:
    print(f"Failed to load '{module_name}' using importlib: {e}")
    def compute_indicators(*args, **kwargs): raise RuntimeError("axist_technical not loaded due to error")
    def prepare_features(*args, **kwargs): raise RuntimeError("axist_technical not loaded due to error")
    def generate_signal_output(*args, **kwargs): raise RuntimeError("axist_technical not loaded due to error")
    from colorama import Fore, Style


# Mock XGBoost model for generate_signal_output
class MockModel:
    def predict_proba(self, X):
        if hasattr(self, 'mock_proba_dist'):
            return np.array([self.mock_proba_dist])
        return np.array([[0.1, 0.1, 0.8]])

class TestComputeIndicators(unittest.TestCase):
    def setUp(self):
        data_size = 250
        self.df = pd.DataFrame({
            'Open': np.random.rand(data_size) * 100 + 100,
            'High': np.random.rand(data_size) * 100 + 105,
            'Low': np.random.rand(data_size) * 100 + 95,
            'Close': np.random.rand(data_size) * 100 + 100,
            'Volume': np.random.rand(data_size) * 1000 + 100
        })
        self.df['High'] = self.df[['Open', 'High', 'Close']].max(axis=1)
        self.df['Low'] = self.df[['Open', 'Low', 'Close']].min(axis=1)
        self.df.index = pd.to_datetime([pd.Timestamp('2023-01-01') + pd.Timedelta(days=i) for i in range(data_size)])

    def test_compute_daily_indicators(self):
        self.assertTrue(functions_loaded, "Module axist_technical not loaded")
        timeframe = 'daily'
        df_processed = compute_indicators(self.df.copy(), timeframe=timeframe)
        self.assertIn(f'EMA20_{timeframe}', df_processed.columns)
        self.assertIn(f'EMA100_{timeframe}', df_processed.columns)
        self.assertIn(f'SUPERT_{timeframe}', df_processed.columns)
        self.assertIn(f'SUPERTd_{timeframe}', df_processed.columns)
        self.assertIn(f'DCL_{timeframe}', df_processed.columns)
        supert_d_values = df_processed[f'SUPERTd_{timeframe}'].dropna()
        if not supert_d_values.empty:
            self.assertTrue(all(val in [-1, 1] for val in supert_d_values))
        dcl_values = df_processed[f'DCL_{timeframe}'].dropna()
        dcm_values = df_processed[f'DCM_{timeframe}'].dropna()
        dcu_values = df_processed[f'DCU_{timeframe}'].dropna()
        common_indices = dcl_values.index.intersection(dcm_values.index).intersection(dcu_values.index)
        if not common_indices.empty:
            self.assertTrue(all(dcl_values[idx] <= dcm_values[idx] for idx in common_indices))
            self.assertTrue(all(dcm_values[idx] <= dcu_values[idx] for idx in common_indices))

    def test_compute_hourly_indicators(self):
        self.assertTrue(functions_loaded, "Module axist_technical not loaded")
        timeframe = 'hourly'
        df_processed = compute_indicators(self.df.copy(), timeframe=timeframe)
        self.assertIn(f'EMA20_{timeframe}', df_processed.columns)
        self.assertIn(f'DCL_{timeframe}', df_processed.columns)
        if not df_processed[f'DCL_{timeframe}'].dropna().empty:
             self.assertFalse(df_processed[f'DCL_{timeframe}'].isna().all())

    def test_compute_5m_indicators(self):
        self.assertTrue(functions_loaded, "Module axist_technical not loaded")
        timeframe = '5m'
        df_processed = compute_indicators(self.df.copy(), timeframe=timeframe)
        self.assertIn(f'EMA20_{timeframe}', df_processed.columns)
        self.assertIn(f'DCL_{timeframe}', df_processed.columns)
        self.assertTrue(df_processed[f'DCL_{timeframe}'].isna().all())

class TestPrepareFeatures(unittest.TestCase):
    def setUp(self):
        self.assertTrue(functions_loaded, "Module axist_technical not loaded for TestPrepareFeatures")
        data_size = 250
        self.df_1d = pd.DataFrame({
            'Open': np.random.rand(data_size) * 100 + 100, 'High': np.random.rand(data_size) * 100 + 105,
            'Low': np.random.rand(data_size) * 100 + 95, 'Close': np.random.rand(data_size) * 100 + 100,
            'Volume': np.random.rand(data_size) * 1000 + 100
        }, index=pd.to_datetime([pd.Timestamp('2023-01-01') + pd.Timedelta(days=i) for i in range(data_size)]))
        self.df_1d['High'] = self.df_1d[['Open', 'High', 'Close']].max(axis=1)
        self.df_1d['Low'] = self.df_1d[['Open', 'Low', 'Close']].min(axis=1)

        intraday_size = data_size * 2
        self.df_1h = self.df_1d.iloc[:max(1,data_size//24)+1].resample('1H').ffill().iloc[:intraday_size//4] if data_size//24 > 0 else pd.DataFrame()
        self.df_5m = self.df_1d.iloc[:max(1,data_size//(24*12))+1].resample('5min').ffill().iloc[:intraday_size] if data_size//(24*12) > 0 else pd.DataFrame()
        self.df_30m = self.df_1d.iloc[:max(1,data_size//(24*2))+1].resample('30min').ffill().iloc[:intraday_size//2] if data_size//(24*2) > 0 else pd.DataFrame()
        self.df_90m = self.df_1d.iloc[:max(1,data_size//(24//2))+1].resample('90min').ffill().iloc[:intraday_size//3] if data_size//(24//2) > 0 else pd.DataFrame()

    def test_prepare_features_columns(self):
        self.assertTrue(functions_loaded, "Module axist_technical not loaded")
        if self.df_5m.empty or self.df_30m.empty or self.df_1h.empty or self.df_90m.empty or self.df_1d.empty:
            self.skipTest("One or more sample intraday DataFrames are empty in setUp for TestPrepareFeatures.")
            return
        features_df = prepare_features(
            self.df_5m.copy(), self.df_30m.copy(), self.df_1h.copy(),
            self.df_90m.copy(), self.df_1d.copy(), horizon=5, drop_recent=False
        )
        if features_df.empty:
            self.fail("features_df is empty after prepare_features call.")
        self.assertIn('EMA20_daily', features_df.columns)
        self.assertIn('DCL_1h', features_df.columns)

class TestGenerateSignalOutput(unittest.TestCase):
    def setUp(self):
        self.assertTrue(functions_loaded, "Module axist_technical not loaded for TestGenerateSignalOutput")
        self.mock_model = MockModel()
        self.ticker = "TEST"
        self.base_latest_row = pd.Series({
            'Close': 100.0, 'EMA20_daily': 95.0, 'EMA50_daily': 90.0,
            'EMA100_daily': 85.0, 'EMA200_daily': 80.0,
            'SUPERTd_daily': 1.0, 'ATR_daily': 2.0,
            'DCL_daily': 90.0, 'DCM_daily': 95.0, 'DCU_daily': 100.0
        })

    def test_strong_uptrend_long_signal(self):
        self.mock_model.mock_proba_dist = [0.1, 0.1, 0.8]
        latest_row = self.base_latest_row.copy()
        signal = generate_signal_output(self.ticker, latest_row, self.mock_model, 0.03)
        self.assertIsNotNone(signal)
        self.assertIn("Strong Uptrend (P>E20>E50, E50>E100>E200, ST_Up)", signal)

    def test_strong_downtrend_short_signal(self):
        self.mock_model.mock_proba_dist = [0.8, 0.1, 0.1]
        latest_row = self.base_latest_row.copy()
        latest_row['Close'] = 75.0; latest_row['EMA20_daily'] = 80.0; latest_row['EMA50_daily'] = 85.0
        latest_row['EMA100_daily'] = 90.0; latest_row['EMA200_daily'] = 95.0
        latest_row['SUPERTd_daily'] = -1.0
        signal = generate_signal_output(self.ticker, latest_row, self.mock_model, 0.03)
        self.assertIsNotNone(signal)
        self.assertIn("Strong Downtrend (P<E20<E50, E50<E100<E200, ST_Down)", signal)

    def test_weak_uptrend_signal(self):
        self.mock_model.mock_proba_dist = [0.1, 0.1, 0.8]
        latest_row = self.base_latest_row.copy()
        latest_row['EMA100_daily'] = 70.0; latest_row['SUPERTd_daily'] = np.nan
        signal = generate_signal_output(self.ticker, latest_row, self.mock_model, 0.03)
        self.assertIsNotNone(signal)
        self.assertIn("Weak Uptrend (P>E20>E50)", signal)

    def test_neutral_choppy_signal(self):
        self.mock_model.mock_proba_dist = [0.1, 0.1, 0.8]
        latest_row = self.base_latest_row.copy()
        latest_row['EMA20_daily'] = 102.0; latest_row['EMA50_daily'] = 105.0
        latest_row['EMA100_daily'] = 90.0; latest_row['EMA200_daily'] = 95.0
        latest_row['SUPERTd_daily'] = np.nan
        signal = generate_signal_output(self.ticker, latest_row, self.mock_model, 0.03)
        self.assertIsNotNone(signal); self.assertIn("Neutral/Choppy", signal)
        if "(" in signal and "()" not in signal : self.assertIn("(", signal)
        else: self.assertNotIn("(",signal)

    def test_no_signal_low_probability(self):
        self.mock_model.mock_proba_dist = [0.4, 0.3, 0.3]
        latest_row = self.base_latest_row.copy()
        signal = generate_signal_output(self.ticker, latest_row, self.mock_model, 0.03)
        self.assertIsNone(signal)

    def test_no_signal_neutral_class(self):
        self.mock_model.mock_proba_dist = [0.3, 0.4, 0.3]
        latest_row = self.base_latest_row.copy()
        signal = generate_signal_output(self.ticker, latest_row, self.mock_model, 0.03)
        self.assertIsNone(signal)

    def test_atr_fallback(self):
        self.mock_model.mock_proba_dist = [0.1, 0.1, 0.8]
        latest_row = self.base_latest_row.copy()
        latest_row['ATR_daily'] = np.nan
        signal = generate_signal_output(self.ticker, latest_row, self.mock_model, 0.03)
        self.assertIsNotNone(signal)
        self.assertIn("(est. ATR)", signal)

if __name__ == '__main__':
    unittest.main()
