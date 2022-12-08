import pandas as pd

def book_name_input(all_book):
    book_name = input("Название книги: ")

    return book_name

def grade_input():
    grade = int(input("Оценка: "))

    return grade


def creating_user_book_list(all_book):
    print("Введите названиея книг и их оценки.")
    print("#### Для выхода введите exit на месте первого аругмента.###")

    user_book_list = []
    grade_list = []

    while True:
        end_of_input = input()
        if end_of_input == "exit": break

        book_name = book_name_input(all_book)      
        grade = grade_input()      

        user_book_list.append(book_name)
        grade_list.append(grade)

    y = pd.DataFrame({'id_book': user_book_list,
                      'user': -1,
                      'grade': grade_list})

    return y
