from datetime import datetime, timezone, timedelta


yesterday_end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
yesterday_start = yesterday_end - timedelta(days=1)

print(yesterday_start)
print(yesterday_end)

text = "can you delete clean my house in the todo list?"
date_keywords = ['add', 'update', 'delete']
res = any(keyword in text.lower().split() for keyword in date_keywords)
print(res)
