def equalizeArray(arr):
    keeper = [0] * 101
    for num in arr:
        keeper[num] += 1
    return len(arr) - max(keeper)

def queensAttack(n, k, r_q, c_q, obstacles):
    leftmax = 0
    rightmin = n + 1
    bottommax = 0
    topmin = n + 1
    top_left = min(n - r_q, c_q - 1)  
    top_right = min(n - r_q, n - c_q)
    bottom_right = min(r_q - 1, n - c_q)
    bottom_left = min(r_q - 1, c_q - 1)
    direction_4 = bottom_right + 1
    direction_7 = bottom_left + 1
    direction_11 = top_left + 1
    direction_1 = top_right + 1
    directions = 0
    counter = 0
    for o in obstacles:
        if o[0] == r_q and o[1] < c_q and o[1] > leftmax:
            leftmax = o[1]
            continue
        elif o[0] == r_q and o[1] > c_q and o[1] < rightmin:
            rightmin = o[1]
            continue
        elif o[1] == c_q and o[0] < r_q and o[0] > bottommax:
            bottommax = o[0]
            continue
        elif o[1] == c_q and o[0] > r_q and o[0] < topmin:
            topmin = o[0]
            continue
        elif r_q - o[0] > 0 and r_q - o[0] == o[1] - c_q and r_q - o[0] < direction_4:
            direction_4 = r_q - o[0]
            continue
        elif r_q - o[0] > 0 and r_q - o[0] == c_q - o[1] and r_q - o[0] < direction_7:
            direction_7 = r_q - o[0]
            continue
        elif o[0] - r_q > 0 and o[0] - r_q == c_q - o[1] and o[0] - r_q < direction_11:
            direction_11 = o[0] - r_q
            continue
        elif o[0] - r_q > 0 and o[0] - r_q == o[1] - c_q and o[0] - r_q < direction_1:
            direction_1 = o[0] - r_q
            continue
    
    directions += direction_1 + direction_4 + direction_7 + direction_11 - 4
    counter = (c_q - leftmax - 1) + (rightmin - c_q - 1) + (r_q - bottommax - 1) + (topmin - r_q - 1) + directions

    return counter

def acmTeam(topics):
    topic_length = len(topics[0])
    max_counter = 0
    num_of_way = 0
    counter_list = []

    for _i in range(len(topics) - 1):
        counter = 0
        for _j in range(_i + 1, len(topics)):
            or_str = int(topics[_i], 2) | int(topics[_j], 2) 
            c = bin(or_str)[2:].count("1")
            if c >= max_counter:
                max_counter = c
                counter_list.append(c)
    
    for way in counter_list:
        if way == max_counter:
            num_of_way += 1

    return [max_counter, num_of_way]
    
def taumBday(b, w, bc, wc, z):
    total_cost = 0
    
    if bc + z < wc and bc < wc:
        total_cost = ((b + w) * bc) + (w * z)
    elif wc + z < bc and wc < bc:
        total_cost = ((b + w) * wc) + (b * z)
    else:
        total_cost = b * bc + w * wc
    
    return total_cost

def organizingContainers(container):
    sumOfContainers = [0] * len(container) 
    sumOfTypes = [0] * len(container) 
    for i, c in enumerate(container):
        sumOfContainers[i] += sum(c)
        for j, v in enumerate(c):
            sumOfTypes[j] += v
    
    sumOfContainers.sort()
    sumOfTypes.sort()

    return "Possible" if sumOfContainers == sumOfTypes else "Impossible"

import math

def encryption(s):
    s = s.replace(" ", "")
    _row = math.floor(math.sqrt(len(s)))
    _col = _row + 1 if len(s) / _row != _row else _row
    _keep_index = 0
    res = ""

    if _row * (_row + 1) < len(s):
        _row += 1
        _col = _row
    
    _matrix = [[0] * _col for _i in range(_row)]   
    
    for _i in range(_row):
        for _j in range(_col):
            if _keep_index < len(s):
                _matrix[_i][_j] = s[_keep_index]
                _keep_index += 1

    for _i in range(_col):
        for row in _matrix:
            if row[_i] != 0:
                res += row[_i]
        res += " "
    
    return res

def biggerIsGreater(w):
    _l = len(w)
    swapper = 0
    compare_list = []
    mutated_w = list(w)
    
    for _i in reversed(range(_l)):
        for _j in reversed(range(_i)):
            if w[_i] > w[_j]:
                if _j >= swapper:
                    compare_list.append((_i, _j, _i - _j))
                    swapper = _j
                else:
                    break
    
    if compare_list == []:
        return "no answer"

    sorted_compare_list = sorted(compare_list, key=lambda k: k[1], reverse=True)
    check = sorted_compare_list[0]
    final_list = list(filter(lambda x: x[1] == check[1] and w[x[0]] < w[check[0]], sorted_compare_list))
    
    if final_list != []:
        final = final_list[0]
    else:
        final = sorted_compare_list[0]
        
    temp = mutated_w[final[0]]
    mutated_w[final[0]] = mutated_w[final[1]]
    mutated_w[final[1]] = temp
    return "".join(mutated_w[:final[1] + 1] + sorted(mutated_w[final[1] + 1:]))

def kaprekarNumbers(p, q):
    arr = []
    
    for _i in range(p, q + 1):
        sqrt = _i * _i
        split_index = len(str(sqrt)) // 2
        if split_index == 0:
            if _i == sqrt:
                arr.append(_i)
        else:
            left = str(sqrt)[:split_index]
            right = str(sqrt)[split_index:]
            _sum = int(left) + int(right)
            if _sum == _i:
                arr.append(_i)
    if arr == []:
        print("INVALID RANGE")
    else:
        for x in arr:
            print(x, end=" ")

def beautifulTriplets(d, arr):
    l = len(arr)
    total_triplets = 0
    for _i in range(l - 1):
        triplet_complete = 1
        _j = _i + 1
        coeff = 1
        while _j < l:
            if arr[_j] - arr[_i] == coeff * d:
                coeff += 1
                triplet_complete += 1
            if triplet_complete == 3:
                total_triplets += 1
                break
            _j += 1
    return total_triplets

def minimumDistances(arr):
    minDist = math.inf
    for _i in range(len(arr)):
        for _j in range(_i + 1, len(arr)):
            if arr[_i] == arr[_j] and _j - _i < minDist:
                minDist = _j - _i
    if minDist == math.inf:
        return -1
    else:
        return minDist
    
def howManyGames(p, d, m, s):
    g = p
    counter = 0
    while s >= g:
        counter += 1
        s -= g
        if g - d > m:
            g = g - d
        else:
            g = m
    return counter

print(howManyGames(16, 2, 1, 9981))