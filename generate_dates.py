import datetime
from dateutil.relativedelta import relativedelta

# Calculate yesterday's date
end_date_yesterday = datetime.date.today() - datetime.timedelta(days=1)
end_date_str = end_date_yesterday.strftime("%Y-%m-%d")

# Calculate start date for 3 months ago
start_date_3mo = end_date_yesterday - relativedelta(months=3)
start_date_3mo_str = start_date_3mo.strftime("%Y-%m-%d")

# Calculate start date for 1 month ago
start_date_1mo = end_date_yesterday - relativedelta(months=1)
start_date_1mo_str = start_date_1mo.strftime("%Y-%m-%d")

print(f"YESTERDAY_DATE={end_date_str}")
print(f"START_DATE_3MO={start_date_3mo_str}")
print(f"START_DATE_1MO={start_date_1mo_str}")
