from sympy import Symbol, Function, Matrix, zeros, symbols, N, sqrt, Pow, Mul, Add, log, exp, sin, cos, tan
import sympy

const_lookup = {}
coords = []
dcoords = []

def set_coords(_coords, _dcoords):
    for c in _coords:
        coords.append(c)
    for dc in _dcoords:
        dcoords.append(dc)

def sym_formatter(expr, disable_dereference_check=False, max_iterations=-1):
    is_fully_reduced = False
    orig_expr = expr
    
    expr = _sym_formatter(expr.expand())
    expr = _sym_formatter(expr.simplify())
    
    iterations = 0
    
    while not is_fully_reduced:
        prev_expr = expr
        expr = _sym_formatter(expr).simplify()
        
        if prev_expr == expr:
            is_fully_reduced = True
            
        iterations += 1
        
        if iterations == max_iterations:
            break
            
    # Test whether it dereferences properly
    if not disable_dereference_check:
        diff = (dereference_expr(expr) - orig_expr).simplify()
        if type(expr) == sympy.ImmutableDenseMatrix or type(expr) == Matrix:
            if diff != zeros(*diff.shape):
                raise Exception('Sym formatter failed to dereference properly.')
        else: 
            if (dereference_expr(expr) - orig_expr).simplify() != 0:
                raise Exception('Sym formatter failed to dereference properly.')
            
    return expr

def _sym_formatter(expr):
    expr = sympy.sympify(expr)
    
    if type(expr) == Mul:
        return mul_formatter(expr)
    elif type(expr) == Add:
        return add_formatter(expr)
    elif type(expr) == Pow:
        return pow_formatter(expr)
    elif type(expr) == sympy.ImmutableDenseMatrix or type(expr) == Matrix:
        return matrix_formatter(expr)
    elif type(expr) in [log, exp, tan, cos, sin]:
        return func_formatter(expr, type(expr))
    elif expr in coords or expr in dcoords:
        return expr
    elif expr.is_constant():
        return expr
    elif type(expr) == Symbol:
        return expr
    else:
        print(expr)
        print(expr in [exp])
        print(type(expr))
        raise Exception('Unforseen expr: %s' % expr)
        
def register_const_symbol(const_expr):
    const_expr = sympy.sympify(const_expr)
    
    # Check if negative, in which case remove the negative sign
    if type(const_expr) == Mul and const_expr.args[0].is_constant() and const_expr.args[0] < 0:
        is_negative = True
        const_expr = -const_expr
    else:
        is_negative = False
    
    # See if const_expr is just a one symbol or constant, in which case we just return it
    if type(const_expr) == Symbol or const_expr.is_constant():
        if is_negative:
            return -const_expr
        else:
            return const_expr
    
    # See if const_expr has already been registered
    const_expr_expanded = const_expr.expand()
    #neg_const_expr_expanded = -const_expr_expanded
    for reg_const_k, (reg_const_sym, reg_const_expr) in const_lookup.items():
        diff = reg_const_expr.expand() - const_expr_expanded
        diff = diff.simplify().trigsimp()
        if diff == 0:
            if is_negative:
                return -reg_const_sym
            else:
                return reg_const_sym
        
        #diff = reg_const_expr.expand() - neg_const_expr_expanded
        #diff = diff.simplify().trigsimp()
        #if diff == 0:
        #    return -reg_const_sym
    
    const_expr = const_expr.simplify()
    expr_var_name = expr_to_var_string(const_expr)
    const_expr_symbol = Symbol(expr_var_name)
    const_lookup[expr_var_name] = (const_expr_symbol, const_expr)
    
    if is_negative:
        return -const_expr_symbol
    else:
        return const_expr_symbol

def matrix_formatter(expr):
    return expr.applyfunc(_sym_formatter)

def mul_formatter(expr):
    const_expr = 1
    non_const_expr = 1
    
    # Sort into constant and non-constant expr
    
    for arg in expr.args:
        
        if arg in coords or arg in dcoords:
            non_const_expr *= arg
        elif type(arg) == Symbol:
            const_expr *= arg
        elif arg.is_constant():
            const_expr *= arg
        else:
            non_const_expr *= _sym_formatter(arg)
            
    # Register the const expr in the const_lookup table
    
    const_expr_symbol = register_const_symbol(const_expr)
            
    return const_expr_symbol * non_const_expr

def add_formatter(expr):
    const_expr = 0
    non_const_expr = 0
    
    # Sort into constant and non-constant expr
    
    for arg in expr.args:
        
        if arg in coords or arg in dcoords:
            non_const_expr += arg
        elif type(arg) == Symbol:
            const_expr += arg
        elif arg.is_constant():
            const_expr += arg
        else:
            non_const_expr += _sym_formatter(arg)
            
    # Register the const expr in the const_lookup table
    
    const_expr_symbol = register_const_symbol(const_expr)
            
    return const_expr_symbol + non_const_expr

def func_formatter(expr, func):
    func_arg = expr.args[0]
    func_arg = _sym_formatter(func_arg)
    
    if len(expr.args) > 1:
        raise Exception('Something wrong here...')
        
    if func_arg in coords or func_arg in dcoords:
        return func(func_arg)
    elif type(func_arg) == Symbol or func_arg.is_constant():
        new_expr = register_const_symbol(func(func_arg))
        return new_expr
    else:
        return func(func_arg) # If it reaches this stage we assume nothing else can be done
    #else:
    #    raise Exception('Unforseen expr: %s' % expr)
        
def pow_formatter(expr):
    expr_base, expr_exponent = expr.args
    
    expr_base = _sym_formatter(expr_base)
    expr_exponent = _sym_formatter(expr_exponent)
    
    if (expr_base in coords or expr_base in dcoords) or (expr_exponent in coords or expr_exponent in dcoords):
        new_expr = Pow(expr_base, expr_exponent)
    elif (expr_base.is_constant() or type(expr_base) == Symbol) and (expr_exponent.is_constant() or type(expr_exponent) == Symbol):
        new_expr = register_const_symbol(Pow(expr_base, expr_exponent))
    else:
        new_expr = Pow(expr_base, expr_exponent)
        
    return new_expr

def dereference_expr(expr):
    if type(expr) == sympy.ImmutableDenseMatrix or type(expr) == Matrix:
        return expr.applyfunc(_dereference_expr)
    else:
        return _dereference_expr(expr)

def _dereference_expr(expr):
    expr = sympy.sympify(expr)
    expr_type = type(expr)
    new_args = []
    
    for arg in expr.args:
        if type(arg) == Symbol and arg.name.startswith('var'):
            new_args.append(dereference_expr( const_lookup[arg.name][1] ))
        else:
            new_args.append(dereference_expr( arg ))
            
    if expr.is_constant():
        return expr
    elif expr_type == Symbol:
        if expr.name.startswith('var'):
            return const_lookup[expr.name][1]
        else:
            return expr
    else:
        return expr_type(*new_args)
    
def get_used_vars(exprs):
    if type(exprs) != list:
        exprs = [exprs]
    
    var_list = []
    
    for expr in exprs:
        _get_used_vars(expr, var_list)
        
    var_list = list(set(var_list))
        
    return var_list
    
def _get_used_vars(expr, var_list):
    if type(expr) == sympy.ImmutableDenseMatrix or type(expr) == Matrix:
        expr.applyfunc(lambda expr: _get_used_vars(expr, var_list))
    
    expr = sympy.sympify(expr)
    
    for arg in expr.args:
        if type(arg) == Symbol and arg.name.startswith('var'):
            var_list.append(arg.name)
        else:
            _get_used_vars(arg, var_list)
            
def expr_to_var_string(expr):
    str_expr = str(expr)
    str_expr = str_expr.replace(' ', '')
    str_expr = str_expr.replace('**', '_pow_')
    str_expr = str_expr.replace('*', '_tim_')
    str_expr = str_expr.replace('/', '_div_')
    str_expr = str_expr.replace('-', '_min_')
    str_expr = str_expr.replace('+', '_pls_')
    str_expr = str_expr.replace('(', '_lpar_')
    str_expr = str_expr.replace(')', '_rpar_')
    str_expr = str_expr.replace('.', 'p')
    
    str_expr = 'var_' + str_expr
    return str_expr