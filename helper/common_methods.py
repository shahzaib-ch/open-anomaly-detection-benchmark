import ast


def read_dictionary_from_file(file_path):
    """
    Returns dictionary after reading from file
    """
    with open(file_path, 'r') as f:
        s = f.read()
        return ast.literal_eval(s)


def save_dictionary_to_file(dictionary, file_path):
    """
    Saves dictionary to file
    """
    with open(file_path, 'w') as f:
        f.write(str(dictionary))
