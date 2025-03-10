# Automated paper fetching

## 2025-03-10: update github action

Update the previous server deployment mode to use GitHub Actions.

## 2025-02-11: deployment (deprecated)

​	The automated paper scraping script has been deployed on server 153 using crontab, and it automatically runs every day at 20:20 to fetch papers related to large model inference from the past day. Please avoid manually editing the `daily-arxiv-llm.md` file, as this could cause the content of the file to become disorganized.

```shell
20 20 * * * /bin/bash /home/wyf/paper/paper-update.sh >> /home/wyf/paper/paper-log.txt 2>&1
```

## Website

* [https://papers.cool/arxiv/cs.DC](https://papers.cool/arxiv/cs.DC): Distributed, Parallel, and Cluster Computing

## Configuration

+ python=3.9
+ requests,bs4,pytz

## Usage

+ The script uses two files under the "tools/" directory: `keyword-accept.csv` and `keyword-reject.csv`. These files are used to filter papers based on keywords. You can modify these files to set the filtering strategy. However, you must ensure that the `keyword-reject.csv` file contains at least one keyword, otherwise no papers will be crawled. The reason for this is unclear.

+ Select the time period for scraping papers.

```python
tz = pytz.timezone("Asia/Shanghai")
now = datetime.now(tz)
start_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")  
end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")  
```

You can check the script execution logs in `/home/wyf/paper/log.txt`.

