import many_stop_words

def get_stopwords(language_code: str) -> set[str]:
    try:
        return many_stop_words.get_stop_words(language_code)
    except:
        return set()
        
if __name__ == "__main__":
    print(get_stopwords("en"))
    print(get_stopwords("de"))
    print(get_stopwords("hi"))
    print(get_stopwords("zh"))
    print(get_stopwords("ja"))
    print(get_stopwords("ar"))