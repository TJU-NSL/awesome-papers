import os
import pytz
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

PAPER_FILE_NAME = 'daily-arxiv-llm.md'
WARNING_TEXT = "The paper list will be updated automatically, please do not edit.\n\n"


def read_keywords_from_csv(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip().lower() for line in file.readlines()]


def get_paper_info(paper):
    title_tag = paper.find('a', class_='title-link')
    title = title_tag.text.strip()
    link = "https://arxiv.org" + title_tag['href']
    abs_link = link.replace('/arxiv/', '/abs/')
    return f"* [{title}]({abs_link})"


def fetch_papers(date, accept_keywords, reject_keywords, subjects=["DC", "OS"]):
    papers = []
    for subject in subjects:  
        url = f"https://papers.cool/arxiv/cs.{subject}?date={date}"
        print(url)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data for date: {date}")
            return []
    
        soup = BeautifulSoup(response.content, 'html.parser')
        papers.extend(soup.find_all('div', class_='panel paper'))

    selected_papers = []
    for paper in papers:
        title_tag = paper.find('a', class_='title-link')
        title = title_tag.text.strip()
        link = "https://arxiv.org" + title_tag['href']
        abs_link = link.replace('/arxiv/', '/abs/')
        abstract_tag = paper.find('span', class_='abstract')
        abstract = abstract_tag.text.strip() if abstract_tag else ""

        print(title)

        # keyword filter
        if any(keyword in title.lower() or keyword in abstract.lower() for keyword in accept_keywords):
            if not reject_keywords or not any(
                    exclude_keyword in title.lower() or exclude_keyword in abstract.lower() for exclude_keyword in
                    reject_keywords):
                selected_papers.append(f"* [{title}]({abs_link})")

    return selected_papers


def date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=1)
    current = end
    while current >= start:
        yield current.strftime("%Y-%m-%d")
        current -= delta


def get_start_date():
    # get the last date in the local file
    if not os.path.isfile(PAPER_FILE_NAME):
        return "2025-01-01"

    lines = open(PAPER_FILE_NAME, 'r', encoding='utf-8').readlines()
    for line in lines:
        if line.startswith("### "):
            return (datetime.strptime(line[4:14], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    return "2025-01-01"


def format_papers(papers, date):
    formatted = f"### {date}\n\n"
    formatted += '\n'.join(papers) + '\n'
    return formatted


def update_local_file(all_content):
    if os.path.exists(PAPER_FILE_NAME):
        with open(PAPER_FILE_NAME, 'r', encoding='utf-8') as file:
            current_content = file.read()

        # remove the WARNING_TEXT -> create new file content: WARNING_TEXT + new content + current content
        if current_content.startswith(WARNING_TEXT):
            current_content = current_content[len(WARNING_TEXT):]
        new_content = WARNING_TEXT + all_content + '\n' + current_content
    else:
        new_content = WARNING_TEXT + all_content

    with open(PAPER_FILE_NAME, 'w', encoding='utf-8') as file:
        file.write(new_content)

    print(f"file {PAPER_FILE_NAME} updated successfully")


def main(start, end):
    all_papers_content = ""

    accept_keywords = read_keywords_from_csv('keyword-accept.csv')
    reject_keywords = read_keywords_from_csv('keyword-reject.csv')

    for date in date_range(start, end):
        print(f"Fetching papers for date: {date}")
        papers = fetch_papers(date, accept_keywords, reject_keywords)
        if papers:
            formatted_papers = format_papers(papers, date)
            all_papers_content += formatted_papers + '\n'
        else:
            print(f"No papers found for date: {date}")

    if all_papers_content:
        update_local_file(all_papers_content)
    else:
        print("There are no papers to update.")


if __name__ == "__main__":
    now = datetime.now(pytz.timezone("Asia/Shanghai"))
    start_date = get_start_date()
    end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f'fetch papers from {start_date} to {end_date}')

    main(start_date, end_date)
