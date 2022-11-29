from pathlib import Path
import requests
from fake_useragent import UserAgent
from selectolax.lexbor import LexborHTMLParser
from time import sleep
from tqdm import tqdm

TOP_LINK_PATH = Path("top_link.txt")
USER_CSV_PATH = Path("user.csv")
CSV_HEADER = "book; user; grade; rating_grade; publication_date; comment \n"

# Активация UserAgent
useragent = UserAgent()


def main():
    # Проверка файл на наличие в директории + Добавляет заголовок нашего csv файла, если файл отсутствует.
    if not USER_CSV_PATH.exists():
        create_file_with_header(USER_CSV_PATH)

    with open(USER_CSV_PATH, "a", encoding='utf-8') as file:
        sites = read_link_from_file(TOP_LINK_PATH)

        for url in tqdm(sites):
            html, status_code = get_html(url)
            # Парсинг списка с отзывами
            line = ''.join(score_user(score_link(html, url)))
            # Проверяем пустая ли строка
            if line:
                file.write(line)


# Чтение ссылок на сайты из top_link.txt
def read_link_from_file(path: Path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    book_id = [int(element.strip("'{}")) for element in text.split(", ")]
    return [f"https://fantlab.ru/work{i}" for i in sorted(book_id)]


# Получение html страницы и статуса ответа на запрос.
def get_html(url):
    headers = {"Accept": "*/*", "User-Agent": useragent.random}
    # Устанавливаем постоянное соединение
    session = requests.Session()
    session.headers = headers
    adapter = requests.adapters.HTTPAdapter(pool_connections=100,
                                            pool_maxsize=100)
    session.mount('http://', adapter)
    resp = requests.get(url, headers=headers)
    html = resp.text
    return html, resp.status_code


# Получение ссылок на страницы с отзывами
def score_link(html, url):
    tree = LexborHTMLParser(html)
    tree_users_list = tree.css_first(r'span.page-links')

    link_list = []
    # Пользователи без данного элемента не имеют отзывов
    if tree_users_list is not None:
        tree_users = tree_users_list.css(r'a')
        for user in tree_users:
            # Ссылка на страницу с комментариями
            link = url + user.attributes['href']
            link_list.append(link)
        return link_list
    else:
        link_list.append(url)
        return link_list


# Получение отзывов пользователей
def score_user(links):
    score_list = []

    # Пройтись по ссылкам на страницы с отзывами
    for url in links:
        sleep(0.5)
        html, status_code = get_html(url)
        tree = LexborHTMLParser(html)

        # Проверка ответа сервера
        if status_code != 200:
            print(f'ERROE_{status_code}:{url}')
            sleep(1)
            return

        score = tree.css("div.responses-list > div.response-item")
        if score is not None:
            # Пройтись по отзывам
            for user in score:
                # книга; пользователь; оценка_книги; рейтинг_отзыва; дата_публикации; отзыв
                book_link = url.split('?')[0]
                user_id = user.css_first(
                    r'p.response-autor-info>b>a').attributes['href']
                book_rating = user.css_first(
                    r'div.clearfix>div.response-autor-mark>b>span').text()
                comment_rating = user.css_first(
                    r'div.response-votetab>span:nth-of-type(2)').text()
                data_score = user.css_first(
                    r'p.response-autor-info>span').attributes['content']
                body_score = user.css_first(
                    r'div.response-body-home').text().replace('\n', ' ')
                score_list.append(
                    f'{book_link};{user_id};{book_rating};{comment_rating};{data_score};{body_score}\n'
                )

    return score_list


# Создать заголовок для csv
def create_file_with_header(path: Path) -> None:
    with open(path, "w") as fd:
        fd.write(CSV_HEADER)


if __name__ == '__main__':
    main()