import torch
from pathlib import Path

# Node class for the binary tree
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    def __repr__(self): # only used for printing
        if self.left is None and self.right is None:
            return str(self.value)
        return f"{self.value}({self.left},{self.right})"
 
# take a prefix notation and return it's tree structure (recursively calls from root node)
# note: most special functions end with _1 or _2 to indicate unary or binary operations   
def prefix_to_tree(tokens):
    
    idx = 0
    def _parse():
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("Incomplete prefix expression.")
        token = tokens[idx]
        idx += 1
        node = Node(token)
        # Define operator arities.
        unary_ops = {'[CLS]', 'INT-', 'INT+', 'ln', 'exp', 'abs', 'sqrt', 'Complex', 'Re',
                     'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 
                     'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
                     'acos', 'asin', 'atan', 'acot', 'asec', 'acsc',
                     'acosh', 'asinh', 'atanh', 'acoth', 'asech', 'acsch'}
        binary_ops = {'add', 'sub', 'mul', 'div', 'pow'}
        if token in unary_ops or token.endswith('_1'):
            node.left = _parse()
            node.right = None
        elif token in binary_ops or token.endswith('_2'):
            node.left = _parse()
            node.right = _parse()
        else:
            node.left = node.right = None
        return node
    root = _parse()
    if idx != len(tokens):
        print("ValueError cause by: ", tokens)
        raise ValueError("Extra tokens after complete parsing.")
    return root

def path_to_index(path):
    """
    Convert a tuple representing a path (e.g. (0,1,0)) to an integer by interpreting it as a binary number.
    The first element is the most-significant bit.
    """
    idx = 0
    for bit in path:
        idx = (idx << 1) | bit
    return idx

def get_prefix_data_with_paths(expr):
    """
    Given an expression (a list of tokens), parse it into a tree and traverse it in prefix order.
    For each node, return its token and its path as a tuple (with the root being ()).
    Returns:
      - token_list: list of tokens in prefix order.
      - path_list: list of corresponding paths.
    """
    tree = prefix_to_tree(expr)
    token_list = []
    path_list = []
    def traverse(node, path):
        if node is None:
            return
        token_list.append(node.value)
        path_list.append(path)
        traverse(node.left, path + (0,))
        traverse(node.right, path + (1,))
    traverse(tree, ())
    return token_list, path_list

# given a list of expressions, return a list of those that are invalid
def find_invalid_expressions(expressions):
    invalid_expressions = []
    for expr in expressions:
        try:
            prefix_to_tree(expr)  # Attempt to parse the expression
        except ValueError:
            invalid_expressions.append(expr)  # If ValueError occurs, add to the list
    return invalid_expressions

def push_down(current_encoding, child_index, n=2):
    if child_index >= n:
        raise ValueError(f"child_index out of range for n={n}")
    one_hot = [0] * n
    one_hot[child_index] = 1
    new_encoding = one_hot + current_encoding[: len(current_encoding) - n]
    return new_encoding

def precompute_all_positions(n, k):
    """
    Precompute positional encodings for all paths up to depth k.
    Returns a list `positions` of length (k+1), where positions[L] is a tensor of shape (n**L, d_model).
    """
    d_model = n * k
    positions = []
    positions.append(torch.zeros(1, d_model, dtype=torch.float))  # Level 0: root.
    for L in range(1, k+1):
        level_encodings = []
        for encoding in positions[L-1]:
            base = encoding.tolist()
            for child in range(n):
                new_enc = push_down(base, child, n)
                level_encodings.append(torch.tensor(new_enc, dtype=torch.float))
        positions.append(torch.stack(level_encodings, dim=0))
    return positions

def save_precomputed_positions(precomputed_positions, cfg):
    """
    Save the list of positional-encoding tensors to disk under the data.input_dir folder.
    """
    out_dir = Path(cfg.data.vocab_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "precomputed_positions.pt"
    torch.save(precomputed_positions, save_path)
    print(f"Saved positional encodings to {save_path}")