"""Display-only formatting; numeric source data and exports stay unchanged."""


def percentage_table(frame, columns):
    displayed = frame.copy()
    for name in columns:
        displayed[name] = displayed[name].map(lambda value: f'{value:.2%}')
    return displayed
