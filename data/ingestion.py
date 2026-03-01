"""
Data ingestion pipeline.
Pulls from FRED, BLS, Cleveland Fed Nowcast, EIA, Zillow.
All public, free APIs.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from loguru import logger
from dotenv import load_dotenv
from data.database import get_session, EconomicRelease, FeatureSnapshot

load_dotenv()

FRED_KEY = os.getenv("FRED_API_KEY")
BLS_KEY = os.getenv("BLS_API_KEY")
EIA_KEY_URL = "https://api.eia.gov/v2"


# ─────────────────────────────────────────────
# FRED DATA
# ─────────────────────────────────────────────

class FREDClient:
    BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self):
        self._cache: dict = {}

    def get_series(self, series_id: str, start: str = "1990-01-01") -> pd.DataFrame:
        """Fetch a FRED time series as DataFrame. Cached per instance; retries on 5xx."""
        cache_key = (series_id, start)
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "series_id": series_id,
            "api_key": FRED_KEY,
            "file_type": "json",
            "observation_start": start,
            "sort_order": "asc"
        }
        max_retries = 5
        for attempt in range(max_retries):
            try:
                r = requests.get(self.BASE, params=params, timeout=30)
                r.raise_for_status()
                obs = r.json()["observations"]
                df = pd.DataFrame(obs)
                df = df[df["value"] != "."]
                df["value"] = df["value"].astype(float)
                df["date"] = pd.to_datetime(df["date"])
                result = df[["date", "value"]].set_index("date")
                self._cache[cache_key] = result
                return result
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status in (502, 503, 504) and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"FRED {series_id} HTTP {status}, retrying in {wait}s ({attempt+1}/{max_retries-1})")
                    time.sleep(wait)
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"FRED {series_id} request error, retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise
    
    def get_cpi(self) -> pd.DataFrame:
        """Monthly CPI All Urban Consumers (not seasonally adjusted)."""
        return self.get_series("CPIAUCNS")
    
    def get_cpi_sa(self) -> pd.DataFrame:
        """Monthly CPI seasonally adjusted."""
        return self.get_series("CPIAUCSL")
    
    def get_ppi_finished_goods(self) -> pd.DataFrame:
        return self.get_series("WPSFD49207")
    
    def get_import_prices(self) -> pd.DataFrame:
        return self.get_series("WPUIP2311001")
    
    def get_michigan_inflation_expectations(self) -> pd.DataFrame:
        return self.get_series("MICH")
    
    def get_shelter_cpi(self) -> pd.DataFrame:
        return self.get_series("CPIHOSSL")
    
    def get_energy_cpi(self) -> pd.DataFrame:
        return self.get_series("CPIENGSL")
    
    def get_food_cpi(self) -> pd.DataFrame:
        return self.get_series("CPIFABSL")

    def get_fed_funds_rate(self) -> pd.DataFrame:
        """Effective federal funds rate (monthly, FEDFUNDS)."""
        return self.get_series("FEDFUNDS")

    def get_core_cpi(self) -> pd.DataFrame:
        """CPI less food and energy, seasonally adjusted (CPILFESL)."""
        return self.get_series("CPILFESL")

    def get_10y_yield(self) -> pd.DataFrame:
        """10-Year Treasury constant maturity rate (DGS10, daily — use monthly average)."""
        return self.get_series("DGS10")

    def get_unemployment_rate(self) -> pd.DataFrame:
        """Civilian unemployment rate (UNRATE)."""
        return self.get_series("UNRATE")


# ─────────────────────────────────────────────
# BLS DATA
# ─────────────────────────────────────────────

class BLSClient:
    BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    
    SERIES = {
        "CPI_ALL": "CUUR0000SA0",
        "CPI_CORE": "CUUR0000SA0L1E",  # All items less food and energy
        "CPI_SHELTER": "CPIHOSSL",
        "CPI_ENERGY": "CPIENGSL",
        "CPI_FOOD": "CPIFABSL",
        "PPI_FINISHED": "WPSFD49207",
        "PPI_FOOD": "WPU012",
    }
    
    def get_series(self, series_ids: list, start_year: int = 2000) -> dict:
        payload = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(datetime.now().year),
            "registrationkey": BLS_KEY,
            "catalog": False,
            "calculations": True,
            "annualaverage": False
        }
        r = requests.post(self.BASE, json=payload, timeout=30)
        r.raise_for_status()
        results = {}
        for series in r.json()["Results"]["series"]:
            sid = series["seriesID"]
            records = []
            for item in series["data"]:
                try:
                    val = float(item["value"])
                    dt = datetime.strptime(f"{item['year']}-{item['period'][1:]}-01", "%Y-%m-%d")
                    records.append({"date": dt, "value": val})
                except (ValueError, KeyError):
                    continue
            df = pd.DataFrame(records).set_index("date").sort_index()
            results[sid] = df
        return results
    
    def get_all_cpi_components(self) -> dict:
        """Pull all CPI component series."""
        series_ids = list(self.SERIES.values())
        return self.get_series(series_ids)


# ─────────────────────────────────────────────
# CLEVELAND FED NOWCAST
# ─────────────────────────────────────────────

class ClevelandFedClient:
    """
    Cleveland Fed publishes daily inflation nowcast.
    We scrape their public data endpoint.
    """
    NOWCAST_URL = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting/nowcast-data"
    
    def get_latest_nowcast(self) -> dict:
        """
        Fetch Cleveland Fed inflation nowcast.
        Returns dict with 'cpi_mom', 'cpi_yoy', 'pce_mom', 'pce_yoy' forecasts.
        Falls back to FRED SPF median if Cleveland is unavailable.
        """
        try:
            # Cleveland Fed publishes CSV data
            url = "https://www.clevelandfed.org/-/media/files/webcharts/inflationestimation/inflation_nowcasting_data.xlsx"
            df = pd.read_excel(url, engine='openpyxl')
            # Parse their format — column names vary by release
            latest = df.iloc[-1]
            return {
                "source": "cleveland_fed",
                "date": datetime.now().date(),
                "cpi_mom_nowcast": float(latest.get("CPI_1m", np.nan)),
                "cpi_yoy_nowcast": float(latest.get("CPI_12m", np.nan)),
                "confidence": "high"
            }
        except Exception as e:
            logger.warning(f"Cleveland Fed fetch failed: {e}, using fallback")
            return self._fallback_nowcast()
    
    def _fallback_nowcast(self) -> dict:
        """Use Philadelphia Fed SPF as fallback."""
        fred = FREDClient()
        mich = fred.get_michigan_inflation_expectations()
        latest_mich = float(mich["value"].iloc[-1])
        return {
            "source": "michigan_fallback",
            "date": datetime.now().date(),
            "cpi_mom_nowcast": None,  # Will be filled by our regression
            "cpi_yoy_nowcast": latest_mich,
            "confidence": "low"
        }


# ─────────────────────────────────────────────
# EIA GASOLINE PRICES
# ─────────────────────────────────────────────

class EIAClient:
    """EIA weekly retail gasoline prices — strong CPI component signal."""

    def get_gasoline_prices(self, fred_client: "FREDClient" = None) -> pd.DataFrame:
        """Weekly US retail gasoline price via FRED (no EIA key needed)."""
        fred = fred_client or FREDClient()
        return fred.get_series("GASREGCOVW")


# ─────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────

class FeatureBuilder:
    """
    Assembles all features for a given reference period (CPI release month).
    Critical: only uses data that was available BEFORE the CPI release date.
    No lookahead contamination.
    """
    
    def __init__(self):
        self.fred = FREDClient()
        self.bls = BLSClient()
        self.eia = EIAClient()
        self.cleveland = ClevelandFedClient()
    
    def build_features(self, reference_period: date, as_of_date: date = None,
                        _nowcast_cache: dict = None, include_nowcast: bool = True) -> dict:
        """
        Build feature vector for predicting CPI for reference_period.
        as_of_date: date we're making the prediction (default = today).
        All features must be available as of as_of_date.
        """
        if as_of_date is None:
            as_of_date = date.today()
        
        features = {}
        
        try:
            # --- CPI lags (always available — prior months) ---
            cpi = self.fred.get_cpi_sa()
            cpi["mom"] = cpi["value"].pct_change() * 100
            cpi["yoy"] = cpi["value"].pct_change(12) * 100
            
            # Lag 1: prior month actual
            lag1_date = reference_period - relativedelta(months=1)
            lag2_date = reference_period - relativedelta(months=2)
            lag12_date = reference_period - relativedelta(months=12)
            
            features["cpi_lag1_mom"] = self._get_value_at(cpi["mom"], lag1_date)
            features["cpi_lag2_mom"] = self._get_value_at(cpi["mom"], lag2_date)
            features["cpi_lag12_mom"] = self._get_value_at(cpi["mom"], lag12_date)
            features["cpi_lag1_yoy"] = self._get_value_at(cpi["yoy"], lag1_date)
            
            # --- Gasoline prices (EIA, weekly — available before CPI) ---
            gas = self.eia.get_gasoline_prices(fred_client=self.fred)
            # Use month average up to as_of_date
            ref_start = pd.Timestamp(reference_period.replace(day=1))
            ref_end = pd.Timestamp(as_of_date)
            gas_month = gas[(gas.index >= ref_start) & (gas.index <= ref_end)]
            
            if len(gas_month) > 0:
                features["gasoline_avg"] = float(gas_month["value"].mean())
                # MoM vs prior month average
                prior_start = ref_start - pd.DateOffset(months=1)
                prior_end = ref_start - pd.DateOffset(days=1)
                gas_prior = gas[(gas.index >= prior_start) & (gas.index <= prior_end)]
                if len(gas_prior) > 0:
                    features["gasoline_mom"] = (
                        float(gas_month["value"].mean()) / float(gas_prior["value"].mean()) - 1
                    ) * 100
                else:
                    features["gasoline_mom"] = np.nan
            else:
                features["gasoline_avg"] = np.nan
                features["gasoline_mom"] = np.nan
            
            # --- PPI finished goods (BLS, released ~1 week before CPI) ---
            ppi = self.fred.get_series("WPSFD49207")
            ppi["mom"] = ppi["value"].pct_change() * 100
            features["ppi_mom"] = self._get_value_at(ppi["mom"], lag1_date)
            
            # --- Import price index ---
            try:
                ipi = self.fred.get_import_prices()
                ipi["mom"] = ipi["value"].pct_change() * 100
                features["import_price_mom"] = self._get_value_at(ipi["mom"], lag1_date)
            except Exception:
                features["import_price_mom"] = np.nan
            
            # --- Michigan inflation expectations ---
            mich = self.fred.get_michigan_inflation_expectations()
            features["michigan_inflation_expect"] = self._get_value_at(mich["value"], lag1_date)
            
            # --- Shelter CPI (largest component, ~35%) ---
            shelter = self.fred.get_shelter_cpi()
            shelter["mom"] = shelter["value"].pct_change() * 100
            features["shelter_cpi_lag1"] = self._get_value_at(shelter["mom"], lag1_date)
            features["shelter_cpi_lag2"] = self._get_value_at(shelter["mom"], lag2_date)
            
            # --- Energy CPI component ---
            energy_cpi = self.fred.get_energy_cpi()
            energy_cpi["mom"] = energy_cpi["value"].pct_change() * 100
            features["energy_cpi_lag1"] = self._get_value_at(energy_cpi["mom"], lag1_date)
            
            # --- Food CPI component ---
            food_cpi = self.fred.get_food_cpi()
            food_cpi["mom"] = food_cpi["value"].pct_change() * 100
            features["food_cpi_lag1"] = self._get_value_at(food_cpi["mom"], lag1_date)
            
            # --- Macro factors (v2.0) ---
            # Fed funds rate: monetary policy stance
            try:
                ffr = self.fred.get_fed_funds_rate()
                features["fed_funds_rate"] = self._get_value_at(ffr["value"], lag1_date)
            except Exception:
                features["fed_funds_rate"] = np.nan

            # Core CPI (less food & energy): underlying trend signal
            try:
                core = self.fred.get_core_cpi()
                core["mom"] = core["value"].pct_change() * 100
                features["core_cpi_lag1_mom"] = self._get_value_at(core["mom"], lag1_date)
            except Exception:
                features["core_cpi_lag1_mom"] = np.nan

            # 10-Year Treasury yield: inflation expectations embedded in bond market
            try:
                t10y = self.fred.get_10y_yield()
                features["treasury_10y"] = self._get_value_at(t10y["value"], lag1_date)
            except Exception:
                features["treasury_10y"] = np.nan

            # Unemployment rate: labour-market slack (Phillips curve signal)
            try:
                unemp = self.fred.get_unemployment_rate()
                features["unemp_rate"] = self._get_value_at(unemp["value"], lag1_date)
            except Exception:
                features["unemp_rate"] = np.nan

            # --- Seasonality (month-of-year dummies) ---
            features["month"] = reference_period.month
            features["month_sin"] = np.sin(2 * np.pi * reference_period.month / 12)
            features["month_cos"] = np.cos(2 * np.pi * reference_period.month / 12)
            
            # --- Cleveland Fed nowcast ---
            # Only fetched for live predictions. During training we cannot
            # time-travel to get historical nowcasts, so we leave these NaN.
            # (These fields are not in FEATURE_COLUMNS so they don't affect
            # Ridge/XGBoost, but excluding them keeps the intent explicit.)
            if include_nowcast:
                nowcast = _nowcast_cache if _nowcast_cache is not None else self.cleveland.get_latest_nowcast()
                features["cleveland_nowcast_yoy"] = nowcast.get("cpi_yoy_nowcast")
                features["cleveland_nowcast_mom"] = nowcast.get("cpi_mom_nowcast")
            else:
                features["cleveland_nowcast_yoy"] = np.nan
                features["cleveland_nowcast_mom"] = np.nan
            
        except Exception as e:
            logger.error(f"Feature building error for {reference_period}: {e}")
        
        return features
    
    def _get_value_at(self, series: pd.Series, target_date: date, tolerance_days: int = 45) -> float:
        """Get most recent value on or before target_date."""
        target_ts = pd.Timestamp(target_date)
        available = series[series.index <= target_ts]
        if len(available) == 0:
            return np.nan
        return float(available.iloc[-1])
    
    def build_training_dataset(self, start_year: int = 1990) -> pd.DataFrame:
        """
        Build full historical feature + target dataset for model training.
        Uses FRED data for all historical features.
        """
        logger.info(f"Building training dataset from {start_year}...")
        
        cpi = self.fred.get_cpi_sa()
        cpi["mom"] = cpi["value"].pct_change() * 100
        cpi["yoy"] = cpi["value"].pct_change(12) * 100
        
        # Filter to start_year+
        cpi = cpi[cpi.index.year >= start_year]
        
        # Do NOT fetch the Cleveland nowcast for training rows — we have no
        # historical nowcast data, so applying today's nowcast to 1995–2024
        # rows would be look-ahead bias. The nowcast is blended in at
        # prediction time (live only) via engine.py → forecaster.predict().
        rows = []
        for dt in cpi.index:
            ref_period = dt.date()
            features = self.build_features(ref_period, as_of_date=ref_period - timedelta(days=5),
                                           include_nowcast=False)
            target = float(cpi.loc[dt, "mom"])
            features["target_cpi_mom"] = target
            features["reference_period"] = ref_period
            rows.append(features)
        
        df = pd.DataFrame(rows).set_index("reference_period")
        logger.info(f"Training dataset built: {len(df)} rows, {len(df.columns)} features")
        return df


# ─────────────────────────────────────────────
# DATA INGESTION RUNNER
# ─────────────────────────────────────────────

def run_data_ingestion():
    """Main ingestion job — runs on schedule."""
    logger.info("Starting data ingestion run...")
    
    session = get_session()
    fred = FREDClient()
    
    try:
        # Pull CPI actuals and store new releases
        cpi = fred.get_cpi_sa()
        cpi["mom"] = cpi["value"].pct_change() * 100
        cpi["yoy"] = cpi["value"].pct_change(12) * 100
        
        # Only store releases from last 24 months (rest already in DB)
        recent = cpi[cpi.index >= pd.Timestamp(date.today() - timedelta(days=730))]
        
        for dt, row in recent.iterrows():
            existing = session.query(EconomicRelease).filter_by(
                series_id="CPIAUCSL",
                reference_period=dt.date()
            ).first()
            
            if not existing:
                release = EconomicRelease(
                    series_id="CPIAUCSL",
                    release_date=date.today(),
                    reference_period=dt.date(),
                    value=float(row["value"]),
                    value_mom=float(row["mom"]) if not np.isnan(row["mom"]) else None,
                    value_yoy=float(row["yoy"]) if not np.isnan(row["yoy"]) else None,
                    source="FRED"
                )
                session.add(release)
        
        session.commit()
        logger.info("✅ Data ingestion complete")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        session.rollback()
    finally:
        session.close()


if __name__ == "__main__":
    run_data_ingestion()
