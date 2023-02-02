def GetTextQueries(path: str) -> set:
    queries = []
    with open(path, 'r') as f:
        lines = f.readlines()
        queries = [line.replace('\n', '') for line in lines]
    return queries