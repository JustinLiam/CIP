import covidcast
from contextlib import contextmanager
from datetime import date, timedelta
import os
import signal
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

## BELOW IS THE EXAMPLE CODE - update your own api key
#################################################################
#################################################################

target_counties = [
    '01045',
]

if os.environ.get("EPI_COUNTIES"):
    target_counties = [
        county.strip().zfill(5)
        for county in os.environ["EPI_COUNTIES"].replace(",", " ").split()
        if county.strip()
    ]

# query covidcast API for data
# Note:
#   smoothed_adj_cli is https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/doctor-visits.html
#   wcli (since April 15) from https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/fb-survey.html
#   and many more if we use since 2020-09-08
covidcast.use_api_key(os.environ.get("COVIDCAST_API_KEY"))
start_date, end_date = date(2020, 6, 1), date(2021, 7, 31)


class _CovidcastTimeout(TimeoutError):
    pass


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _api_date(day):
    return day.strftime("%Y%m%d")


@contextmanager
def _time_limit(seconds):
    seconds = int(seconds or 0)
    if seconds <= 0:
        yield
        return

    def _raise_timeout(signum, frame):
        raise _CovidcastTimeout(f"covidcast request exceeded {seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _date_chunks(start, end, chunk_days):
    if int(chunk_days or 0) <= 0:
        yield start, end
        return
    current = start
    delta = timedelta(days=int(chunk_days) - 1)
    while current <= end:
        chunk_end = min(current + delta, end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def _county_batches(counties, batch_size):
    batch_size = int(batch_size or 0)
    if batch_size <= 0 or batch_size >= len(counties):
        yield counties
        return
    for i in range(0, len(counties), batch_size):
        yield counties[i:i + batch_size]


def _fetch_signal_direct_range(signal_name):
    county_batch_size = _env_int("EPI_DELPHI_COUNTY_BATCH_SIZE", 1)
    timeout_seconds = _env_int("EPI_DELPHI_TIMEOUT_SECONDS", 120)
    retries = max(1, _env_int("EPI_DELPHI_RETRIES", 1))
    url = "https://api.delphi.cmu.edu/epidata/api.php"
    frames = []

    for counties in _county_batches(target_counties, county_batch_size):
        params = {
            "source": "covidcast",
            "data_source": "indicator-combination",
            "signals": signal_name,
            "time_type": "day",
            "geo_type": "county",
            "time_values": f"{_api_date(start_date)}-{_api_date(end_date)}",
            "geo_value": ",".join(counties),
        }
        api_key = os.environ.get("COVIDCAST_API_KEY")
        if api_key:
            params["api_key"] = api_key
        for attempt in range(1, retries + 1):
            try:
                print(
                    f"Fetching Delphi direct {signal_name}: {params['time_values']}, "
                    f"counties={params['geo_value']}, attempt={attempt}/{retries}",
                    flush=True,
                )
                t0 = time.time()
                response = requests.get(url, params=params, timeout=timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                message = payload.get("message")
                if message not in {"success", "no results"}:
                    raise RuntimeError(f"Delphi API returned message={message!r}")
                rows = payload.get("epidata") or []
                print(
                    f"Fetched Delphi direct {signal_name}: rows={len(rows)} "
                    f"seconds={time.time() - t0:.2f}",
                    flush=True,
                )
                if rows:
                    frames.append(pd.DataFrame.from_records(rows))
                break
            except Exception as exc:
                print(
                    f"Delphi direct fetch failed for {signal_name} "
                    f"counties={params['geo_value']} attempt={attempt}/{retries}: {exc}",
                    flush=True,
                )
                if attempt >= retries:
                    raise
                time.sleep(min(5 * attempt, 30))

    if not frames:
        raise RuntimeError(f"No Delphi rows returned for {signal_name}")
    out = pd.concat(frames, ignore_index=True)
    out["time_value"] = pd.to_datetime(out["time_value"].astype(str), format="%Y%m%d")
    out["issue"] = pd.to_datetime(out["issue"].astype(str), format="%Y%m%d", errors="coerce")
    out["geo_type"] = "county"
    out["data_source"] = "indicator-combination"
    out["signal"] = signal_name
    return out


def _fetch_signal(signal_name):
    if _env_bool("EPI_DELPHI_DIRECT_RANGE", False):
        return _fetch_signal_direct_range(signal_name)

    chunk_days = _env_int("EPI_DELPHI_CHUNK_DAYS", 0)
    county_batch_size = _env_int("EPI_DELPHI_COUNTY_BATCH_SIZE", 0)
    timeout_seconds = _env_int("EPI_DELPHI_TIMEOUT_SECONDS", 0)
    retries = max(1, _env_int("EPI_DELPHI_RETRIES", 1))

    # Keep the original full-range, all-county request path unless chunking or
    # county batching is explicitly enabled by the caller.
    if chunk_days <= 0 and county_batch_size <= 0 and timeout_seconds <= 0 and retries <= 1:
        return covidcast.signal(
            "indicator-combination",
            signal_name,
            start_date,
            end_date,
            "county",
            geo_values=target_counties
        )

    frames = []
    total_chunks = 0
    for counties in _county_batches(target_counties, county_batch_size):
        for start, end in _date_chunks(start_date, end_date, chunk_days):
            total_chunks += 1
            for attempt in range(1, retries + 1):
                try:
                    print(
                        f"Fetching Delphi {signal_name}: {start}..{end}, "
                        f"counties={','.join(counties)}, attempt={attempt}/{retries}",
                        flush=True,
                    )
                    t0 = time.time()
                    with _time_limit(timeout_seconds):
                        frame = covidcast.signal(
                            "indicator-combination",
                            signal_name,
                            start,
                            end,
                            "county",
                            geo_values=counties
                        )
                    print(
                        f"Fetched Delphi {signal_name}: rows={0 if frame is None else len(frame)} "
                        f"seconds={time.time() - t0:.2f}",
                        flush=True,
                    )
                    if frame is not None and len(frame) > 0:
                        frames.append(frame)
                    break
                except Exception as exc:
                    print(
                        f"Delphi fetch failed for {signal_name} {start}..{end} "
                        f"counties={','.join(counties)} attempt={attempt}/{retries}: {exc}",
                        flush=True,
                    )
                    if attempt >= retries:
                        raise
                    time.sleep(min(5 * attempt, 30))

    if not frames:
        raise RuntimeError(f"No Delphi rows returned for {signal_name} across {total_chunks} chunks")
    return pd.concat(frames, ignore_index=True)


def process_daily_data():
    deaths = _fetch_signal("deaths_incidence_num")
    cases = _fetch_signal("confirmed_incidence_num")

    data = covidcast.aggregate_signals([cases, deaths])

    data = data.rename(
        columns={
            "indicator-combination_confirmed_incidence_num_0_value": "cases",
            "indicator-combination_deaths_incidence_num_1_value": "deaths",
        })

    script_dir = os.path.dirname(__file__)
    project_root = os.path.join(script_dir, "..")
    output_dir = os.path.join(project_root, "data", "delphi_county_data")
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for fips_code in target_counties:
        county_data = data[data['geo_value'] == fips_code][[
            "time_value", "geo_value", "cases", "deaths"
        ]]

        output_file = os.path.join(output_dir, f"{fips_code}_data.csv")
        county_data.to_csv(output_file, index=False)

    print(data[[
        "time_value", "geo_value", "cases", "deaths"
    ]].head(20))

process_daily_data()
