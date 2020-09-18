import re
import inspect
import argparse


def sort_human(list_of_strings):
    """
    Similar to natural sorting
    Code from: https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    """
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*[0-9]*)', key)]
    list_of_strings.sort(key=alphanum)
    return list_of_strings


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')	


def hex_to_bgr(value):
    value = value.lstrip('#')
    lv = len(value)
    rgb = [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
    return rgb[2], rgb[1], rgb[0]


def retrieve_name(var):
    """
    From https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string

    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def count_parameters(model):
    """
    Count the number of parameters in a Torch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
