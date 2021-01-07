import datetime
import time
import requests  # Install this if you don't have it already.


def get_promql_query(metric_name, time_duration, metric_labels):
    query = metric_name
    if metric_labels is not None and len(metric_labels) != 0:
        label_filters = []
        for label_key, label_value in metric_labels.items():
            label_filters.append('{0}="{1}"'.format(label_key, label_value))
        query += '{' + ','.join(label_filters) + '}'
    if time_duration is not None:
        query += '[{0}]'.format(time_duration)
    return query


PROMETHEUS = 'http://localhost:9090/'

# Midnight at the end of the previous month.
end_of_month = datetime.datetime.today().replace(day=1).date()
# Last day of the previous month.
last_day = end_of_month - datetime.timedelta(days=1)
duration = '[' + str(last_day.day) + 'd]'

promql_query = get_promql_query('go_gc_duration_seconds', None, {'quantile': '0', 'job': 'prometheus'})

response = requests.get(PROMETHEUS + '/api/v1/query',
                        params={'query': 'avg(' + promql_query + ')'})
results = response.json()

print(results)
