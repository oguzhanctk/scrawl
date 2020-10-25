import math

def kangaroo(x1, v1, x2, v2):
    for _i in range(10000):
        x1 += v1
        x2 += v2
        if(x1 == x2):
            print("YES")
            return
        else:
            pass
    print("NO")
    return 
    
def fibonacci():
    fib_arr = [1,2]
    limit = 4000000
    index = 1
    while fib_arr[len(fib_arr) - 1] < limit:
        fib_arr.append(fib_arr[index] + fib_arr[index - 1])
        index += 1
    return fib_arr

def getTotalX(a, b):
    arr_a = []
    arr_b = []
    for _i in range(max(a), min(b) + 1):
        counter = 0
        for member_a in a:
            if _i % member_a == 0:
                counter += 1
            else:
                pass
        if counter == len(a):
            arr_a.append(_i)
    for _i in range(max(a), min(b) + 1):
        counter = 0
        for member_b in b:
            if member_b % _i == 0:
                counter += 1
            else:
                pass
        if counter == len(b):
            arr_b.append(_i)
    return len(set(arr_a) & set(arr_b))

def primeFactors(n):
    factors = []
    for _i in range(1, n + 1):
        factor_counter = 0
        if n % _i == 0:
            for _j in range(1, _i + 1):
                if _i % _j == 0:
                    factor_counter += 1
            if factor_counter == 2:
                factors.append(_i)
            else:
                pass
    return factors
    
def breakingRecords(scores_arr):
    min_score = scores_arr[0]
    max_score = scores_arr[0]
    min_counter = 0
    max_counter = 0
    for member in scores_arr:
        if member < min_score:
            min_score = member
            min_counter += 1
        if member > max_score:
            max_score = member
            max_counter += 1
    return([max_counter, min_counter])

def birthday(s, d, m):
    done = 0
    for _i in range(len(s) - m + 1):
        sum = 0
        for _j in range(_i, (_i + m)):
            sum += s[_j]
        if sum == d:
            done += 1
    return done

def divisibleSumPairs(n, k, ar): # n -> len(arr)
    counter = 0
    for _i in range(len(ar)):
        for _j in range(_i + 1, len(ar)):
            if _i < _j and (ar[_i] + ar[_j]) % k == 0:
                counter += 1
    return counter

def migratoryBirds(arr):
    pointer = 0
    res = 0
    birds_arr = [0, 0, 0, 0, 0]
    for member in arr:
        if member == 1:
            birds_arr[0] += 1
        if member == 2:
            birds_arr[1] += 1
        if member == 3:
            birds_arr[2] += 1
        if member == 4:
            birds_arr[3] += 1
        if member == 5:
            birds_arr[4] += 1
    print(birds_arr)
    for _i in range(len(birds_arr)):
        if birds_arr[_i] > res:
            res = birds_arr[_i]
            pointer = _i + 1
    if pointer == 0:
        return 1
    else:
        return pointer

def sockMerchant(n, ar):
    repeated = []
    socks = 0
    for _i in range(n):
        counter = 1
        for _j in range(_i+1, n):
            if ar[_i] == ar[_j] and ar[_i] not in repeated:
                counter += 1
        if counter > 1:
            repeated.append(ar[_i])
        socks += counter // 2
    return socks

def nonDivisibleSubset(k, s): #integer k, integer array s
    if k == 1:
        return 1
    
    res = 0
    freq = [0] * k    
    
    #load remainder frequencies
    for index, value in enumerate(s):
        freq[value % k] += 1

    if freq[0] > 0:
        res += 1
    for _i in range(1, k // 2):
        res += max(freq[_i], freq[k - _i])
    if k % 2 == 0:
        res += 1 if freq[k // 2] else 0
    else:
        res += max(freq[k // 2], freq[k // 2 + 1])

    return res

def repeatedStrings(s, n):
    repeat, modulo, counter = n // len(s), n % len(s), 0
    a_repeat_index = []

    for index, letter in enumerate(s):
        if letter == "a":
            counter += 1
            a_repeat_index.append(index + 1)
    
    res = counter * repeat
    for i in a_repeat_index:
        if i <= modulo:
            res += 1 

    return res
	
def jumpingOnClouds(c):
    loc = 0
    counter = 0
    while loc < len(c):
        if loc == len(c) - 1 and c[loc] == 0:
            return counter
        elif loc == len(c) - 2:
            return counter + 1
        elif c[loc + 2] != 1:
            loc += 2
        else:
            loc += 1
        counter += 1
		
def hurdleRace(k, height):
    max_diff = 0
    for c, value in enumerate(height):
        if (k < value) and (value - k > max_diff):
            max_diff = value - k
    return max_diff

def designerPdfViewer(h, word):
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    max_h = 0
    for c, letter in enumerate(alphabet):
        if letter in word and h[c] > max_h:
            max_h = h[c]
    return max_h * len(word)

def utopianTree(n):
    tree = 1
    for _i in range(0, n):
        if _i % 2 == 0:
            tree *= 2
        else:
            tree += 1
    return tree

def angryProfessor(k, a):
    cancel_counter = 0
    for c, value in enumerate(a):
        if value > 0:
            cancel_counter += 1
        if cancel_counter > len(a) - k:
            return "YES"
    return "NO"

def beautifulDays(i, j, k):
    counter = 0
    for _i in range(i, j + 1):
        if abs(_i - int(str(_i)[::-1])) % k == 0:
            counter += 1
    return counter

def dayOfProgrammer(year):
    dop = None
    if year <= 1917: #julian calendar
        if year % 4 == 0: #leap year
            dop = "12.09." + str(year)
        else:
            dop = "13.09." + str(year) 
    elif year == 1918: #special condition
        dop = "26.09.1918" 
    else: #gregoryen calendar
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            dop = "12.09." + str(year)
        else:
            dop = "13.09." + str(year)
    return dop

def bonAppetit(bill, k, b):
    b_actual = (sum(bill) - bill[k]) / 2 
    if abs(b - b_actual) == 0:
        print("Bon Appetit")
    else:
        print(int(abs(b - b_actual)))

def viralAdvertising(n):
    shared = 5
    liked = 2
    cumulative = 2
    for _i in range(2, n + 1):
        shared = liked * 3
        liked = shared // 2
        cumulative += shared // 2
    return cumulative

def saveThePrisoner(n, m, s):
    pointer = (s + m - 1) % n
    if pointer == 0:
        return n
    else:
        return pointer

def circularArrayRotation(a, k, queries):
    res = []
    #find first index position after rotation
    first_pos = k % len(a)
    #reposition rest of indexes
    temp = [0] * len(a)
    temp[first_pos] = a[0]
    for _i in range(1, len(a)):
        temp[(first_pos + _i) % len(a)] = a[_i]
    for c, value in enumerate(queries):
        res.append(temp[value])
    return res

p = [5, 2, 1, 3, 4]

def permutationEquation(p):
    arr = [0] * len(p)
    temp = None
    for c, value in enumerate(p):
        temp = p[p[value - 1] - 1]
        arr[temp - 1] = value
    return arr

def jumpingOnClouds(c, k):
    isStart = False
    i = 0
    e = 100
    while isStart != True:
        if c[(i + k) % len(c)] == 0:
            e -= 1
        else:
            e -= 3
        i = (i + k) % len(c)
        if i == 0:
            isStart = True
    return e

def findDigits(n):
    counter = 0
    for digit in str(n):
        if int(digit) == 0:
            continue
        elif n % int(digit) == 0:
            counter += 1
    return counter
def squares(a, b):
    start = 1
    arr = []
    while start * start <= b:
        if start * start >= a:
            arr.append(start * start)
        start += 1
    return len(arr)

def libraryFine(d1, m1, y1, d2, m2, y2):
    
    if y2 - y1 < 0:
        return 10000
    elif y2 - y1 > 0:
        return 0
    elif m2 - m1 < 0:
        return 500 * (m1 - m2)
    elif m2 - m1 > 0:
        return 0
    elif d2 - d1 < 0:
        return 15 * (d1 - d2)
    else:
        return 0

def cutTheSticks(sticks):
    counter = None
    result = []
    while len(sticks) > 0:
        m = min(sticks)
        counter = 0
        for c, stick in enumerate(sticks):  
                sticks[c] -= m
                counter += 1
        sticks = list(filter(lambda stick: stick > 0, sticks))                
        result.append(counter)
    return result
    
def bubbleSort(arr):
    r = len(arr)
    for _i in range(r):
        for _j in range(r - 1 - _i):
            if arr[_j] > arr[_j + 1]:
               arr[_j], arr[_j + 1] = arr[_j + 1], arr[_j]
    return arr

def insertionSort(arr):
    r = len(arr)
    for _i in range(1, r):
        key = arr[_i]
        _j = _i - 1
        while _j >= 0 and arr[_j] > key:
            arr[_j + 1] = arr[_j]
            _j -= 1
        arr[_j + 1] = key
    return arr

def quickSort(arr):
    
    if len(arr) <= 1:
        return arr

    less = []
    more = []
    p_itself = []
    pivot = arr[0]

    for _i in range(len(arr)):
        if arr[_i] > pivot:
            more.append(arr[_i])
        elif arr[_i] < pivot:
            less.append(arr[_i])
        else:
            p_itself.append(pivot)
    
    less = quickSort(less)
    more = quickSort(more)
    
    return less + p_itself + more

def mergeSort(arr):
    def merge(left, right):
        left_idx = 0    
        right_idx = 0
        res = []
        while left_idx < len(left) and right_idx < len(right):
            if left[left_idx] < right[right_idx]:
                res.append(left[left_idx])
                left_idx += 1
            else: 
                res.append(right[right_idx])
                right_idx += 1
        if left[left_idx:]:
            res.extend(left[left_idx:])
        if right[right_idx:]:
            res.extend(right[right_idx:])
        return res

    if len(arr) <= 1:
        return arr
    
    middle = len(arr) // 2   

    left = mergeSort(arr[:middle])
    right = mergeSort(arr[middle:])
    
    return merge(left, right)

def insertionSort1(n, arr):
    key = arr[n - 1]
    _j = n - 2
    while _j >= 0 and arr[_j] > key:
        arr[_j + 1] = arr[_j]
        _j -= 1
        print(" ".join(map(lambda x: str(x), arr)))
    arr[_j + 1] = key
    print(" ".join(map(lambda x: str(x), arr)))
    return None

def insertionSort2(n, arr):
    for _i in range(1, n):
        key = arr[_i]
        _j = _i - 1
        while _j >= 0 and arr[_j] > key:
            arr[_j + 1] = arr[_j]
            _j -= 1
        arr[_j + 1] = key
        print(" ".join(map(lambda x: str(x), arr)))
    return None

def countSort(arr):
    r = len(arr)
    C = [[] for _i in range(100)]
    arr = list(map(lambda x: [int(x[0]), x[1]], arr))
    for _i in range(r // 2):
        arr[_i][1] = "-"
    for v in arr:
        C[v[0]].append(v[1])
    C = list(filter(None, C)) 
    return C

def closestNumbers(arr):
    def quickSort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
        less = []
        more = []
        p_itself = []
        for v in arr:
            if v > pivot:
                more.append(v)
            elif v < pivot:
                less.append(v)
            else:
                p_itself.append(v)
        less = quickSort(less)
        more = quickSort(more)
        return less + p_itself + more

    arr = quickSort(arr)
    min_diff = arr[1] - arr[0]
    diff_list = []
    for _i in range(1, len(arr)):
        diff = arr[_i] - arr[_i - 1]
        if diff == min_diff:
            diff_list.extend([arr[_i - 1], arr[_i]])
            min_diff = diff 
        elif diff < min_diff:
            min_diff = diff
            diff_list = []
            diff_list.extend([arr[_i - 1], arr[_i]])
    return diff_list

def findSwap(arr): #find insertion sort swap operations in time less than O(n)^2 
    
    if len(arr) <= 1:
        return 0 
    
    swap = 0

    middle = len(arr) // 2
    left = arr[:middle]
    right = arr[middle:]

    swap += findSwap(left)
    swap += findSwap(right)
    

    left_idx = 0
    right_idx = 0
    k = 0

    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] <= right[right_idx]:
            arr[k] = left[left_idx]
            left_idx += 1
        else:
            arr[k] = right[right_idx]
            right_idx += 1
            swap += len(left) - left_idx
        k += 1
    while left_idx < len(left):
        arr[k] = left[left_idx]
        left_idx += 1
        k += 1
    while right_idx < len(right):
        arr[k] = right[right_idx]
        right_idx += 1
        k += 1
    print(arr)
    return swap

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

def workbook(n, k, arr):
    page_c = 1
    res = 0
    for x in arr:
        flag = 1
        while x > 0:
            if x >= k:
                temp = list(range(flag, flag + k))
            else:
                temp = list(range(flag, flag + x))
            if page_c in temp:
                res += 1
            page_c += 1
            flag += k
            x -= k
    return res
            
def LPS(s): #it will be optimized
    l = len(s)
    res_even = 0
    res_odd = 1
    for _i in range(l - 1):
        #calculate repeated char and extend it  
        if s[_i] == s[_i + 1]:
            left = _i - 1
            right = _i + 2
            len_str = 2
            while left >= 0 and right < l:
                    if s[left] == s[right]:
                        len_str += 2
                        left -= 1
                        right += 1
                    else:
                        break
            if len_str > res_even:
                res_even = len_str
        #calculate odd char and extend it 
        left = _i - 1
        right = _i + 1
        len_str = 1
        while left >= 0 and right < l:
            if s[left] == s[right]:
                len_str += 2
                left -= 1
                right += 1
            else:
                break
        if len_str > res_odd:
            res_odd = len_str
            
    return res_odd if res_even < res_odd else res_even

def flatlandSpaceStations(n, c):
    c.sort()
    start_dist = c[0] - 0
    end_dist = n - 1 - c[-1]
    dist = start_dist if start_dist > end_dist else end_dist
    for _i in range(len(c) - 1):
        _dist = abs(c[_i + 1] - c[_i])
        if _dist // 2 > dist:
            dist = _dist // 2
    return dist

def FairRations(arr):
    odd_counter = 0
    pair_list = []
    bread = 0
    for i, v in enumerate(arr):
        if v % 2 == 1:
            odd_counter += 1
            pair_list.append(i)
    if odd_counter % 2 == 1:
        return "NO"
    
    _i = 0
    while _i  < len(pair_list) - 1:
        bread += 2 * (pair_list[_i + 1] - pair_list[_i])
        _i += 2

    return bread

EDUREKA_VIDEO_TIME = "4:08:20"

def cavityMap(grid):
    edge = len(grid)
    cavity_list = []
    for _i in range(1, edge - 1):
        for _j in range(1, edge - 1):
            if int(grid[_i][_j]) > int(grid[_i + 1][_j]) and \
            int(grid[_i][_j]) > int(grid[_i + -1][_j]) and \
            int(grid[_i][_j]) > int(grid[_i][_j + 1]) and \
            int(grid[_i][_j]) > int(grid[_i][_j - 1]):
                cavity_list.append((_i, _j))
    for cavity in cavity_list:
        temp = [char for char in grid[cavity[0]]]
        temp[cavity[1]] = "X"
        temp = "".join(temp)
        grid[cavity[0]] = temp
    return grid

def stones(n, a, b):
    prob_list = []
    for _i in range(n):
        res = (a * _i) + (b * (n - _i - 1)) 
        prob_list.append(res)
    return sorted(list(set(prob_list)))

def gridSearch(G, P):
    window = None
    grid_row_len = len(G)
    grid_col_len = len(G[0])
    pat_row_len = len(P)
    pat_col_len = len(P[0])    
    for _i in range(grid_row_len - pat_row_len + 1):
        for _j in range(grid_col_len - pat_col_len + 1):
            p_counter = 0
            window = G[_i][_j:_j + pat_col_len]
            if P[p_counter] == window:
                while (p_counter + 1) < pat_row_len and G[_i + p_counter + 1][_j:_j + pat_col_len] == P[p_counter + 1]:
                    p_counter += 1
                if p_counter == (pat_row_len - 1):
                    return "YES"
    return "NO"

def happyLadybugs(b):
    char_list = []
    index_list = []
    underscore_c = 0
    for _i in range(len(b)):
        if b[_i] == "_":
            underscore_c += 1
        elif b[_i] not in char_list:        
            char_list.append(b[_i])
            index_list.append(_i)
        else:
            pass
    for color in char_list:
        if b.count(color) > 1:
            continue
        else:
            return "NO"
    if underscore_c == 0:
        for _i in range(len(index_list) - 1):
            if (index_list[_i + 1] - index_list[_i]) == 1:
                return "NO"
    return "YES"

def strangeCounter(t):
    time = 3
    init = 3
    while time < t:
        time += init * 2
        init *= 2
    value = 1 + (time - t)
    return value

def surfaceArea(A):
    height = len(A)
    width = len(A[0])
    res = 0
    fill_row = [0 for _i in range(width + 2)]

    #fill around with 0 
    for row in A:
        row.insert(0, 0)
        row.append(0)
    A.insert(0, fill_row)
    A.append(fill_row)

    for _i in range(height + 1):
        for _j in range(width + 1):
            res += abs(A[_i][_j] - A[_i + 1][_j])
            res += abs(A[_i][_j] - A[_i][_j + 1])

    return res + (width * height * 2) 

def absolutePermutation(n, k):
    flag = 0
    res_list = []
    range_list = list(range(1, n + 1))
    
    for _i in range(1, n + 1):
        for x in range_list:
            if abs(x - _i) == k:
                res_list.append(x)
                range_list.remove(x)
                break
    
    return res_list if len(res_list) == n else [-1]

def optimal_absolutePermutation(n, k):
    res_list = []

    if k == 0:
        return list(range(1, n + 1))
    
    if (n % (k * 2)) != 0:
        return [-1]
    
    for _i in range(1, n, 2 * k):
        temp = list(range(_i + k, _i + 2 * k)) + list(range(_i, _i + k))
        res_list += temp
    
    return res_list

def bomberMan(n, grid):
    row = len(grid)
    col = len(grid[0])
    bomb_list = []
    final = None

    if n == 0 or n == 1:
        return grid
    
    if n % 2 == 0:
        return ["O" * col for _i in range(row)]

    full_bomb = [(_i, _j) for _j in range(col) for _i in range(row)]
    
    for _i in range(row):
        grid[_i] = list(grid[_i])
        for _j in range(col):
            if grid[_i][_j] == "O":
                bomb_list.append((_i, _j))

    def updateBombList(bomb_list):
        xi = [0, 0, 0, -1, 1]
        yi = [0, 1, -1, 0, 0]
        detonate_list = []
        for bomb in bomb_list:
            for i, j in zip(xi, yi):
                if 0 <= bomb[0] + i <= row - 1 and 0 <= bomb[1] + j <= col - 1:
                    temp = (bomb[0] + i, bomb[1] + j)
                    if temp not in detonate_list:
                        detonate_list.append(temp)
        return set(full_bomb) - set(detonate_list)

    if (n + 1) % 4 == 0:
        final = updateBombList(bomb_list)
    else:
        final = updateBombList(updateBombList(bomb_list))
        

    final_grid = [["." for _j in range(col)] for _i in range(row)]

    for bomb in final:
        final_grid[bomb[0]][bomb[1]] = "O"
    
    return ["".join(row) for row in final_grid]
            

def optimizedBomberMan(n, grid):
    r = len(grid) 
    c = len(grid[0])

    if n == 0 or n == 1:
        return grid
    
    if n % 2 == 0:
        return ["O" * c for _i in range(r)]

    xi = [0, 0, 0, 1, -1]
    yi = [0, 1, -1, 0, 0]
    
    def updateGrid(grid):
        newGrid = [["O"] * c for _i in range(r)]
        
        for _i in range(r):
            for _j in range(c):
                if grid[_i][_j] == "O":
                    for i, j in zip(xi, yi):
                        if 0 <= _i + i < r and 0 <= _j + j < c:
                            newGrid[_i + i][_j + j] = "."
        return newGrid
    
    if (n + 1) % 4 == 0:
        return ["".join(row) for row in updateGrid(grid)]
    else:
        for _i in range(2):
            return ["".join(row) for row in updateGrid(updateGrid(grid))]

def twoPluses(grid):
    r = len(grid)
    c = len(grid[0])
    o = []
    
    #find all validate pluses
    for _i in range(r):
        for _j in range(c):
            plus_size = 0
            if grid[_i][_j] == "G":
                xi = [0, 0, -1, 1]
                yi = [1, -1, 0, 0]
                flag = True
                while flag:
                    num = 0
                    for x, y in zip(xi, yi):
                        if (0 <= (_i + x) < r) and (0 <= (_j + y) < c):
                            if grid[_i + x][_j + y] == "G":
                                num += 1
                            else:
                                break
                    if num == 4:
                        plus_size += 1
                        xi[2] -= 1
                        xi[3] += 1
                        yi[0] += 1
                        yi[1] -= 1
                        o.append((_i, _j, plus_size))
                    else:
                        flag = False
    if len(o) == 0:
        return 1
    
    #overlapping check
    res = o[0][2] * 4 + 1
    for _i in range(len(o) - 1):
        for _j in range(_i + 1, len(o)):
            distx = abs(o[_i][1] - o[_j][1]) - 1 
            disty = abs(o[_i][0] - o[_j][0]) - 1
            calc = 0
            if (o[_i][0] == o[_j][0] or o[_i][1] == o[_j][1]):
                if (distx >= (o[_i][2] + o[_j][2]) or disty >= (o[_i][2] + o[_j][2])):
                    calc = (o[_i][2] * 4 + 1) * (o[_j][2] * 4 + 1)
            else:
                bigger, smaller = (o[_i][2], o[_j][2]) if o[_i][2] >= o[_j][2] else (o[_j][2], o[_i][2])
                if distx >= bigger or disty >= bigger or (distx >= smaller and disty >= smaller):                
                    calc = (o[_i][2] * 4 + 1) * (o[_j][2] * 4 + 1)
            if calc > res:
                print(_i, _j)
                res = calc
    return res
            
def larrysArray(arr):    
    
    def shift(triplet, r):
        for _i in range(r):
            triplet = triplet[1:] + triplet[:1] 
        return triplet
            
    for _i in range(len(arr) - 2):
        print(_i)
        if arr[_i] == _i + 1:
            continue
        for _j in range(_i, len(arr) - 2):
            while _i + 1 not in arr[_j:_j + 3]:
                _j += 1
            for _k in reversed(range(_i, _j + 1)):
                if arr[_i] == _i + 1:
                    break
                temp = arr[_k:_k + 3]
                if temp.index(_i + 1) == 1:
                    temp = shift(temp, 1)
                else:  
                    temp = shift(temp, 2)
                arr[_k:_k + 3] = temp
            break

    if arr[-2] > arr[-1]:
        return "NO"
    else:
        return "YES"

def almostSorted(arr):
    diff = []
    temp_arr = sorted(arr)
    
    for _i in range(len(arr)):
        if arr[_i] != temp_arr[_i]:
            diff.append(_i + 1)
        
    if len(diff) == 0:
        print("yes")
        return 0
    
    if len(diff) == 2:
        print(f"yes\nswap {diff[0]} {diff[1]}")
        return 0
    
    else:
        c = 0
        for _i in range(len(diff) - 1):
            if diff[_i + 1] - diff[_i] != 1:
                if diff[_i + 1] - diff[_i] == 2 and arr[diff[_i]] == temp_arr[diff[_i]]:
                    c += 1
                else:
                    break
            else:
                c += 1
        if c == len(diff) - 1:
            # print(diff)
            k = 0
            for _i in range(len(diff) - 1):
                if arr[diff[_i] - 1] < arr[diff[_i + 1] - 1]:
                    print("no")
                    return 0
            print(f"yes\nreverse {diff[0]} {diff[-1]}")
            return 0
        else:
            print("no")
            return 0

def matrixRotation(matrix, rot):
    final_list = []
    flag = False
    r, c = len(matrix), len(matrix[0])

    if min(r, c) == r:
        final_list = matrix.copy()
    else:
        flag = True
        for _i in range(c):
            temp = []
            for _j in range(r):
                temp.append(matrix[_j][_i])
            final_list.insert(0, temp)

    ref_r, ref_c = (len(final_list), len(final_list[0]))

    rotate_list = []
    for _i in range(ref_r // 2): #extract all rotation layers as array 
        layer = []   
        layer.extend(final_list[ref_r - _i - 1][_i:ref_c - _i])
        
        temp = []
        for _j in reversed(range(_i + 1, ref_r - _i - 1)):
            temp.append(final_list[_j][ref_c - _i - 1])
        layer.extend(temp)
        
        layer.extend(list(reversed(final_list[_i][_i:ref_c - _i])))
        
        temp = []
        for _j in range(_i + 1, ref_r - _i - 1):
            temp.append(final_list[_j][_i])
        layer.extend(temp)
        rotate_list.append(layer) #rotate_list created to keep all layers that will rotate
    # print(rotate_list)
    def rotate(arr, r):
        rot = (r % len(arr))
        if rot == 0:
            return arr
        else:
            t = arr[-rot:] + arr[:-rot]
            return t

    for _i in range(len(rotate_list)):
        rotate_list[_i] = rotate(rotate_list[_i], rot) #rotation completed

    #convert layers to original list template
    for _i in range(ref_r // 2):
        
        final_list[ref_r - _i - 1][_i:ref_c - _i] = rotate_list[_i][:ref_c - (2 * _i)]
        
        _p = 2
        for _j in range(ref_c - (2 * _i), ref_c - (2 * _i) + ref_r - (2 * (_i + 1))):
            final_list[ref_r - _p - _i][ref_c - _i - 1] = rotate_list[_i][_j]
            _p += 1

        pointer = len(rotate_list[_i]) // 2
        
        final_list[_i][_i:ref_c - _i] = reversed(rotate_list[_i][pointer:pointer + ref_c - (2 * _i)])

        _p = 1
        for _j in range(len(rotate_list[_i]) - (ref_r - (2 * (_i + 1))), len(rotate_list[_i])):
            final_list[_p + _i][_i] = rotate_list[_i][_j]
            _p += 1
    
    if flag == True:
        temp = []
        for _i in range(r):
            f = []
            for _j in range(c):
                f.insert(0, final_list[_j][_i])
            temp.append(f)
    else:
        temp = final_list
    
    for row in temp:
        print(" ".join([str(v) for v in row]))
    
    return temp

def bigSorting(arr):
    
    if len(arr) == 1:
        return arr
    
    def merge(left, right):
        left_idx, right_idx = 0, 0
        result = []
        while left_idx < len(left) and right_idx < len(right):
            if len(left[left_idx]) < len(right[right_idx]):
                result.append(left[left_idx])
                left_idx += 1
            elif len(left[left_idx]) == len(right[right_idx]):
                if left[left_idx] <= right[right_idx]:
                    result.append(left[left_idx])
                    left_idx += 1
                else:
                    result.append(right[right_idx])
                    right_idx += 1
            else:
                result.append(right[right_idx])
                right_idx += 1

        if left[left_idx:]:
            result.extend(left[left_idx:])
        if right[right_idx:]:
            result.extend(right[right_idx:])
        return result
    
    middle = len(arr) // 2

    left = arr[:middle]
    right = arr[middle:]

    left = bigSorting(left)
    right = bigSorting(right)

    return merge(left, right)







