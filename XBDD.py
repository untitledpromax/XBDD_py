import numpy as np
import copy
from graphviz import Digraph
from IPython.display import Image
from weakref import WeakValueDictionary

CACHE_SLOTS = 262144
CACHE_HARD_LIMIT = 8388608
CACHE_HIT_THRESHOLD = 0.30
CACHE_MIN_LOOKUPS_BEFORE_RESIZE = 1024


class ComputedTableCache:
    def __init__(self, initial_size=CACHE_SLOTS,
                 hard_limit=CACHE_HARD_LIMIT,
                 hit_threshold=CACHE_HIT_THRESHOLD,
                 min_lookups_before_resize=CACHE_MIN_LOOKUPS_BEFORE_RESIZE):
        self._initial_size = initial_size
        self._max_entries = initial_size
        self._hard_limit = hard_limit
        self._hit_threshold = hit_threshold
        self._min_lookups_before_resize = min_lookups_before_resize
        self._entries = dict()
        self._lookups = 0
        self._hits = 0
        self._insertions = 0
        self._evictions = 0

    def lookup(self, key):
        self._lookups += 1
        if key in self._entries:
            self._hits += 1
            return self._entries[key]
        self._maybe_resize()
        return None

    def insert(self, key, value):
        if key not in self._entries and len(self._entries) >= self._max_entries:
            # Capacity is full: evict the oldest entry in FIFO order
            oldest_key = next(iter(self._entries))
            del self._entries[oldest_key]
            self._evictions += 1
        self._entries[key] = value
        self._insertions += 1

    def _maybe_resize(self):
        if self._lookups < self._min_lookups_before_resize:
            return
        if self._max_entries >= self._hard_limit:
            return
        if self._hits / self._lookups <= self._hit_threshold:
            return
        new_size = min(self._max_entries * 2, self._hard_limit)
        if new_size <= self._max_entries:
            return
        self._max_entries = new_size
        self._lookups = 0
        self._hits = 0

    def clear(self):
        self._entries.clear()
        self._max_entries = self._initial_size
        self._lookups = 0
        self._hits = 0
        self._insertions = 0
        self._evictions = 0

    def stats(self):
        hit_rate = (self._hits / self._lookups) if self._lookups > 0 else 0.0
        return {
            'lookups': self._lookups,
            'hits': self._hits,
            'hit_rate': hit_rate,
            'insertions': self._insertions,
            'evictions': self._evictions,
            'size': len(self._entries),
            'capacity': self._max_entries,
            'hard_limit': self._hard_limit,
        }

    def __contains__(self, key):
        return key in self._entries

    def __getitem__(self, key):
        return self.lookup(key)

    def __setitem__(self, key, value):
        self.insert(key, value)

    def __len__(self):
        return len(self._entries)


"""Define global variables"""
computed_table = ComputedTableCache()
unique_table = WeakValueDictionary()
global_var_order = dict()
global_node_id=0




class Node:
    def __init__(self,key):
        self.id = 0
        self.ref = 0
        self.key = key
        self.out_maps=[]
        self.successor=[None]*2
        self.out_negs=[False]*2
        
class XBDD:
    def __init__(self,node):
        self.neg = False
        self.map = set()
        self.key_2_var = dict()
        self.node=node
            
    def node_number(self):
        node_set=set()
        node_set=get_node_set(self.node,node_set)
        return len(node_set)
    
    def self_copy(self):
        temp = XBDD(self.node)
        temp.neg = self.neg
        temp.map = copy.copy(self.map)
        temp.key_2_var = copy.copy(self.key_2_var)
        return temp
    
    def show(self,real_label=True,file_name='output'):
        succ=[]
        dot=Digraph(name='reduced_tree')
        dot=layout(self.node,dot,succ,real_label)
        dot.node('-0','',shape='none')
        if len(self.map)>0:
            label1=str(self.map)
        else:
            label1 = ''
        if self.neg:
            label1+='o'
        dot.edge('-0',str(self.node.id),color="blue",label=label1)
        dot.format = 'png'
        return Image(dot.render(file_name))
    
    def __eq__(self,other):
        if self.node.id==other.node.id and self.neg==other.neg and self.map==other.map:
            return True
        else:
            return False

        
def layout(node,dot=Digraph(),succ=[],real_label=True):
    col=['red','blue']
    dot.node(str(node.id), str(node.key), fontname="helvetica",shape="circle",color="red")
    for k in range(2):
        if node.successor[k]:
            if len(node.out_maps[k])>0:
                label1=str(node.out_maps[k])
            else:
                label1 = ''
            if node.out_negs[k]:
                label1+='o'
            if not node.successor[k] in succ:
                dot=layout(node.successor[k],dot,succ,real_label)
                dot.edge(str(node.id),str(node.successor[k].id),color=col[k],label=label1)
                succ.append(node.successor[k])
            else:
                dot.edge(str(node.id),str(node.successor[k].id),color=col[k],label=label1)
    return dot        

        
def Ini_XBDD(var_order=[]):
    """To initialize the unique_table,computed_table and set up a global index order"""
    global computed_table
    global unique_table
    global global_node_id
    global_node_id=0
    unique_table = WeakValueDictionary()
    computed_table.clear()
    set_index_order(var_order)
    return get_const_dd()

def Clear_TDD():
    """To initialize the unique_table,computed_table and set up a global index order"""
    global computed_table
    global unique_table
    global global_node_id
    # global add_find_time,add_hit_time,cont_find_time,cont_hit_time
    global_node_id=0
    unique_table.clear()
    computed_table.clear()
    # add_find_time=0
    # add_hit_time=0
    # cont_find_time=0
    # cont_hit_time=0

def get_const_dd():
    node = Find_Or_Add_Unique_table(-1)
    dd = XBDD(node)
    return dd

def get_unique_table():
    return unique_table

def get_unique_table_num():
    return len(unique_table)

def get_cache_stats():
    return computed_table.stats()

def set_index_order(var_order):
    global global_var_order
    global_var_order=dict()
    if isinstance(var_order,list):
        for k in range(len(var_order)):
            global_var_order[var_order[k]]=k
    if isinstance(var_order,dict):
        global_var_order = copy.copy(var_order)
    global_var_order[-1] = float('inf')
    
def get_var_order():
    global global_var_order
    return copy.copy(global_var_order)

def get_node_set(node,node_set=set()):
    """Only been used when counting the node number of a TDD"""
    if not node in node_set:
        node_set.add(node)
        for k in range(2):
            if node.successor[k]:
                node_set = get_node_set(node.successor[k],node_set)
    return node_set

def Find_Or_Add_Unique_table(x,succ_nodes=[],the_map2=set(),negs=[False,False]):
    """To return a node if it already exist, creates a new node otherwise"""
    global global_node_id,unique_table
    
    if x==-1:
        if unique_table.__contains__(x):
            return unique_table[x]
        else:
            res=Node(x)
            res.id=0
            unique_table[x]=res
        return res
    temp_key=[x]+succ_nodes+negs+sorted(the_map2)
    temp_key=tuple(temp_key)

    if temp_key in unique_table:
        return unique_table[temp_key]
    else:
        res=Node(x)
        global_node_id+=1
        res.id=global_node_id
        res.successor=succ_nodes
        res.out_maps = [set(),the_map2]
        res.out_negs = negs
        unique_table[temp_key]=res
    return res

def map_XOR(map1,map2):
    return map1 ^ map2

def normalize(x,the_successors):
    """The normalize and reduce procedure"""

    if the_successors[0]==the_successors[1]:
        return the_successors[0]
    flip=0
    neg = False
    if the_successors[0].node.id>the_successors[1].node.id:
        flip=1
    elif the_successors[0].node.id==the_successors[1].node.id:
        if tuple(sorted(the_successors[0].map))>tuple(sorted(the_successors[1].map)):
            flip=1
    if flip:
        the_successors = [the_successors[1],the_successors[0]]
        
    if the_successors[0].neg:
        neg = True
        the_successors[0].neg = 1-the_successors[0].neg
        the_successors[1].neg = 1-the_successors[1].neg

    the_map2 = the_successors[0].map^the_successors[1].map
    
    succ_nodes=[succ.node for succ in the_successors]
    succ_negs = [succ.neg for succ in the_successors]
    node=Find_Or_Add_Unique_table(x,succ_nodes,the_map2,succ_negs)
    res=XBDD(node)
    res.map = copy.copy(the_successors[0].map)
    if flip:
        res.map.add(x)
    res.neg = neg
    return res

def get_count():
    global add_find_time,add_hit_time,cont_find_time,cont_hit_time,add_time,cont_time
    print(add_time,cont_time)
    print("add:",add_hit_time,'/',add_find_time,'/',add_hit_time/(add_find_time+1))
    print("cont:",cont_hit_time,"/",cont_find_time,"/",cont_hit_time/(cont_find_time+1))

def get_dd(U,var=[]):
    var_sort=sorted(var, key=lambda k: global_var_order[k])
    var_sort.reverse()
    key_2_var = {k:var_sort[k] for k in range(len(var_sort))}
    var_2_key = {v: k for k, v in key_2_var.items()}
    order=[]
    for v in var:
        order.append(var_2_key[v])
    dd = np_2_dd(U,order)
    dd.key_2_var = key_2_var
    return dd

def np_2_dd(U,order=[]):
    #index is the index_set as the axis order of the matrix
    U_dim=U.ndim
    U_shape=U.shape
    if sum(U_shape)==U_dim:
        node=Find_Or_Add_Unique_table(-1)
        res=XBDD(node)
        for k in range(U_dim):
            U=U[0]
        if U==0:
            res.neg = True
        return res
    if not order:
        order=list(range(U_dim))
    x=max(order)
    split_pos=order.index(x)
    order[split_pos]=-1
    split_U=np.split(U,U_shape[split_pos],split_pos)
    the_successors=[]
    for k in range(U_shape[split_pos]):
        res=np_2_dd(split_U[k],copy.copy(order))
        the_successors.append(res)
    tdd = normalize(x,the_successors)
    return tdd
    
def cofactor(dd,x,c):
    """Slice a XBDD with respect to x=c"""

    k=dd.node.key
    
    if k==-1:
        return dd.self_copy()
    
    if k<x:
        res = dd.self_copy()
        res.map.discard(x)
        return res
    
    if k==x:
        if k not in dd.map:
            res=XBDD(dd.node.successor[c])
            res.neg = (dd.neg!=dd.node.out_negs[c])
            res.map = dd.map^dd.node.out_maps[c]
        else:
            res=XBDD(dd.node.successor[1-c])
            res.neg = (dd.neg!=dd.node.out_negs[1-c])
            res.map = dd.map^dd.node.out_maps[1-c]
        res.map.discard(x)
        return res
    else:
        print("Not supported yet!")
        print(dd.node.key,x,c)       

def is_true(dd: 'XBDD'):
    return dd.node.key == -1 and not dd.neg

def is_false(dd: 'XBDD'):
    return dd.node.key == -1 and dd.neg

def dd_hash(dd: 'XBDD'):
    return (dd.node.id, dd.neg, tuple(sorted(list(dd.map))))

def find_ite_cache(F: 'XBDD', G: 'XBDD', H: 'XBDD'):
    cache_key = ('ite', dd_hash(F), dd_hash(G), dd_hash(H))
    return computed_table.lookup(cache_key)

def insert_ite_cache(F: 'XBDD', G: 'XBDD', H: 'XBDD', res: 'XBDD'):
    cache_key = ('ite', dd_hash(F), dd_hash(G), dd_hash(H))
    computed_table.insert(cache_key, res)

def if_then_else(F: 'XBDD', G: 'XBDD', H: 'XBDD'):
    if is_true(F): return G.self_copy()
    if is_false(F): return H.self_copy()
    if G == H: return G.self_copy()
    if is_true(G) and is_false(H): return F.self_copy()
    if is_false(G) and is_true(H):
        res = F.self_copy()
        res.neg = not res.neg
        return res

    dd1 = get_const_dd()
    dd0 = get_const_dd()
    dd0.neg = True

    if F.node.id == G.node.id and F.map == G.map:
        if F.neg == G.neg:
            # ITE(F, F, H) => F + H => ITE(F, 1, H)
            return if_then_else(F, dd1, H)
        else:
            # ITE(F, ~F, H) => ~F * H => ITE(F, 0, H)
            return if_then_else(F, dd0, H)

    if F.node.id == H.node.id and F.map == H.map:
        if F.neg == H.neg:
            # ITE(F, G, F) => F * G => ITE(F, G, 0)
            return if_then_else(F, G, dd0)
        else:
            # ITE(F, G, ~F) => G + ~F => ITE(F, G, 1)
            return if_then_else(F, G, dd1)
    cached_res = find_ite_cache(F, G, H)
    if cached_res is not None:
        return cached_res
    keys = [dd.node.key for dd in (F, G, H) if dd.node.key != -1]
    if not keys:
        return G.self_copy() if is_true(F) else H.self_copy()
    x = max(keys)

    F0 = cofactor(F, x, 0)
    F1 = cofactor(F, x, 1)
    G0 = cofactor(G, x, 0)
    G1 = cofactor(G, x, 1)
    H0 = cofactor(H, x, 0)
    H1 = cofactor(H, x, 1)

    E = if_then_else(F0, G0, H0)
    T = if_then_else(F1, G1, H1)
    if T == E:
        return T

    res = normalize(x, [E, T])
    insert_ite_cache(F, G, H, res)
    return res

def apply(F: 'XBDD', G: 'XBDD', op: str):
    """
    Supported ops: 'AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR'
    """
    dd1 = get_const_dd()
    dd0 = get_const_dd()
    dd0.neg = True

    if op == 'AND':
        # F AND G = ITE(F, G, 0)
        return if_then_else(F, G, dd0)
        
    elif op == 'OR':
        # F OR G = ITE(F, 1, G)
        return if_then_else(F, dd1, G)
        
    elif op == 'XOR':
        # F XOR G = ITE(F, ~G, G)
        not_G = G.self_copy()
        not_G.neg = not not_G.neg
        return if_then_else(F, not_G, G)
        
    elif op == 'NAND':
        # NAND(F, G) = ~ITE(F, G, 0)
        res = if_then_else(F, G, dd0)
        res.neg = not res.neg
        return res
        
    elif op == 'NOR':
        # NOR(F, G) = ~ITE(F, 1, G)
        res = if_then_else(F, dd1, G)
        res.neg = not res.neg
        return res
        
    elif op == 'XNOR':
        # XNOR(F, G) = ITE(F, G, ~G)
        not_G = G.self_copy()
        not_G.neg = not not_G.neg
        return if_then_else(F, G, not_G)
        
    else:
        raise ValueError(f"Unsupported operation: {op}")

def get_truth_table(n, tt):
    array = []
    pivot = 1
    for i in range(2 ** n):
        array.append((1 if tt & pivot else 0))
        pivot <<= 1
    return np.array(array).reshape((2,) * n)

def vector_to_tensor(vec: np.ndarray, n: int) -> np.ndarray:
    if vec.ndim != 1:
        raise ValueError("Input must be a one-dimensional vector")
    if vec.size != (1 << n):          # 1 << n is equivalent to 2**n
        raise ValueError(f"Vector length {vec.size} does not equal 2^{n} = {(1 << n)}")
    return vec.reshape((2,) * n)

def tensor_to_vector(tensor: np.ndarray, var_order_s = []) -> np.ndarray:
    if not var_order_s:
        return tensor.reshape(-1)
    else:
        m = len(var_order_s)
        var_order = []
        for k in range(m-1,-1,-1):
            var_order.append('x'+str(k))
        old_to_idx = {lab: idx for idx, lab in enumerate(var_order)}
        order_n = [old_to_idx[lab] for lab in var_order_s]
        return tensor.transpose(order_n).ravel() 
    
def get_DD_from_truth_table(n, tt,var_order_s = []):
    data = get_truth_table(n, tt)
    # print(data)
    var = []
    var_order = []
    for k in range(n-1,-1,-1):
        var.append('x'+str(k))
        var_order.append('x'+str(k))

    if not var_order_s:
        var_order_s=var_order
    #Ini_XBDD(var_order_s)
    return get_dd(data,var),tensor_to_vector(data,var_order_s)

def random_binary_tensor(n: int) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    shape = (2,) * n
    return np.random.randint(0, 2, size=shape, dtype=np.uint8)

def get_ramdom_DD(n,data=np.array([]),var_order_s = [],add_neg=False):
    if sum(data.shape)==0:
        data = random_binary_tensor(n)
    # print(data)
    var = []
    var_order = []
    for k in range(n-1,-1,-1):
        var.append('x'+str(k))
        var_order.append('x'+str(k))
    if not var_order_s:
        var_order_s=var_order

    Ini_XBDD(var_order_s)
    return get_dd(data,var),tensor_to_vector(data,var_order_s)
