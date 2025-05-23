import re, string, calendar
from wikipedia import WikipediaPage
import wikipedia
from bs4 import BeautifulSoup
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from match import match
from typing import List, Callable, Tuple, Any, Match


def get_page_html(title: str) -> str:
    """Gets html of a wikipedia page

    Args:
        title - title of the page

    Returns:
        html of the page
    """
    results = wikipedia.search(title)
    return WikipediaPage(results[0]).html()




def get_first_infobox_text(html: str) -> str:
    """Gets first infobox html from a Wikipedia page (summary box)

    Args:
        html - the full html of the page

    Returns:
        html of just the first infobox
    """
    soup = BeautifulSoup(html, "html.parser")
    results = soup.find_all(class_="infobox")

    if not results:
        raise LookupError("Page has no infobox")
    return results[0].text


def clean_text(text: str) -> str:
    """Cleans given text removing non-ASCII characters and duplicate spaces & newlines

    Args:
        text - text to clean

    Returns:
        cleaned text
    """
    only_ascii = "".join([char if char in string.printable else " " for char in text])
    no_dup_spaces = re.sub(" +", " ", only_ascii)
    no_dup_newlines = re.sub("\n+", "\n", no_dup_spaces)
    return no_dup_newlines


def get_match(
    text: str,
    pattern: str,
    error_text: str = "Page doesn't appear to have the property you're expecting",
) -> Match:
    """Finds regex matches for a pattern

    Args:
        text - text to search within
        pattern - pattern to attempt to find within text
        error_text - text to display if pattern fails to match

    Returns:
        text that matches
    """
    p = re.compile(pattern, re.DOTALL | re.IGNORECASE)
    match = p.search(text)

    if not match:
        raise AttributeError(error_text)
    return match


def get_polar_radius(planet_name: str) -> str:
    """Gets the radius of the given planet

    Args:
        planet_name - name of the planet to get radius of

    Returns:
        radius of the given planet
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(planet_name)))
    pattern = r"(?:Polar radius.*?)(?: ?[\d]+ )?(?P<radius>[\d,.]+)(?:.*?)km"
    error_text = "Page infobox has no polar radius information"
    match = get_match(infobox_text, pattern, error_text)

    return match.group("radius")


def get_birth_date(name: str) -> str:
    """Gets birth date of the given person

    Args:
        name - name of the person

    Returns:
        birth date of the given person
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"(?:Born\D*)(?P<birth>\d{4}-\d{2}-\d{2})"
    error_text = (
        "Page infobox has no birth information (at least none in xxxx-xx-xx format)"
    )
    match = get_match(infobox_text, pattern, error_text)

    return match.group("birth")


# below are a set of actions. Each takes a list argument and returns a list of answers
# according to the action and the argument. It is important that each function returns a
# list of the answer(s) and not just the answer itself.

def get_atomic_number(element_name: str) -> str:
    """Gets the atomic number of the given chemical element

    Args:
        element_name - name of the element to get atomic number of

    Returns:
        atomic number of the given element
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(element_name)))
    pattern = r"(?:Atomic number.*?)(?P<number>\d+)"
    error_text = "Page infobox has no atomic number information"
    match = get_match(infobox_text, pattern, error_text)

    return match.group("number")


def atomic_number(matches: List[str]) -> List[str]:
    """Returns atomic number of element in matches

    Args:
        matches - match from pattern of element to find atomic number of

    Returns:
        atomic number of element
    """
    return [get_atomic_number(matches[0])]

def get_period(element_name: str) -> str:
    html = get_page_html(element_name)
    infobox_text = clean_text(get_first_infobox_text(html))

    # Pattern to handle 'Period period 2' or just 'Period 2'
    pattern = r"Period\s+period\s+(\d+)|Period\s+(\d+)"
    error_text = f"Page infobox has no period information for {element_name}"
    match = get_match(infobox_text, pattern, error_text)

    # The match could be in group(1) or group(2) depending on which part matched
    period_num = match.group(1) or match.group(2)

    return f"Period {period_num}"

def period(matches: List[str]) -> List[str]:
    """Returns the period string of the given element."""
    return [get_period(matches[0])]


def get_phase(element_name: str) -> str:
    html = get_page_html(element_name)
    infobox_text = clean_text(get_first_infobox_text(html))
    pattern = r"Phase at STP.*?(solid|liquid|gas|unknown)"
    match_obj = re.search(pattern, infobox_text, re.IGNORECASE | re.DOTALL)
    if not match_obj:
        raise AttributeError(f"No phase info found for {element_name}")
    return match_obj.group(1).capitalize()

def phase(matches: List[str]) -> List[str]:
    return [get_phase(matches[0])]





def birth_date(matches: List[str]) -> List[str]:
    """Returns birth date of named person in matches

    Args:
        matches - match from pattern of person's name to find birth date of

    Returns:
        birth date of named person
    """
    return [get_birth_date(" ".join(matches))]


def polar_radius(matches: List[str]) -> List[str]:
    """Returns polar radius of planet in matches

    Args:
        matches - match from pattern of planet to find polar radius of

    Returns:
        polar radius of planet
    """
    return [get_polar_radius(matches[0])]


# dummy argument is ignored and doesn't matter
def bye_action(dummy: List[str]) -> None:
    raise KeyboardInterrupt


# type aliases to make pa_list type more readable, could also have written:
# pa_list: List[Tuple[List[str], Callable[[List[str]], List[Any]]]] = [...]
Pattern = List[str]
Action = Callable[[List[str]], List[Any]]

# The pattern-action list for the natural language query system. It must be declared
# here, after all of the function definitions
pa_list: List[Tuple[List[str], Callable]] = [
    ("what is the atomic number of %".split(), atomic_number),
    ("atomic number of %".split(), atomic_number),

    ("phase of %".split(), phase),
    ("what is the phase of %".split(), phase),
    ("what is the phase at stp of %".split(), phase),

    ("period of %".split(), period),
    ("what is the period of %".split(), period),

    (["bye"], bye_action),
]




def search_pa_list(src: List[str]) -> List[str]:
    """Takes source, finds matching pattern and calls corresponding action. If it finds
    a match but has no answers it returns ["No answers"]. If it finds no match it
    returns ["I don't understand"].

    Args:
        source - a phrase represented as a list of words (strings)

    Returns:
        a list of answers. Will be ["I don't understand"] if it finds no matches and
        ["No answers"] if it finds a match but no answers
    """
    for pat, act in pa_list:
        mat = match(pat, src)
        if mat is not None:
            answer = act(mat)
            return answer if answer else ["No answers"]

    return ["I don't understand"]


def query_loop() -> None:
    """The simple query loop. The try/except structure is to catch Ctrl-C or Ctrl-D
    characters and exit gracefully"""
    print("Welcome to the periodic table of elements!\n")
    while True:
        try:
            print()
            query = input("Your query? ").replace("?", "").lower().split()
            answers = search_pa_list(query)
            for ans in answers:
                print(ans)

        except (KeyboardInterrupt, EOFError):
            break

    print("\nSo long!\n")


# uncomment the next line once you've implemented everything are ready to try it out
query_loop()

