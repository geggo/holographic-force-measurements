
#Padarn Wilson
#https://gist.github.com/e7fbba34cc88e173c9d9.git
#http://stackoverflow.com/questions/9242733/find-the-smallest-regular-number-that-is-not-less-than-n


# TODO: fails for target = np.int64(2103) !??
#import next_regular
#import numpy as np
#next_regular.next_regular_mul8(np.int64(2103))

def next_regular(target:int) -> int:
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
 
    Target must be a positive integer.
    """
    # TODO: np.int64 ????
    target = int(target)  #TODO safe?
    assert isinstance(target, int)

    if target <= 6:
        return target
 
    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target
 
    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2 ** ((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2 ** (len(bin(quotient - 2)) - 2)
 
            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match
 
def next_regular_mul8(x):
   return next_regular(-(-x//8))*8