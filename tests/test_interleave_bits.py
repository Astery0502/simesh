def hex_to_binary_and_find_all_non_zero_positions(hex_number):
    # Convert the hexadecimal number to an integer
    num = int(hex_number, 16)
    
    # Convert the integer to a binary string, removing the '0b' prefix
    binary_string = bin(num)[2:]
    
    # Pad the binary string with leading zeros to make it a full 16-bit representation
    binary_string = binary_string.zfill(16)
    
    # Find all positions of non-zero bits
    non_zero_positions = [i for i, bit in enumerate(binary_string) if bit == '1']
    
    return binary_string, non_zero_positions

def interleave_bits(ign):
    answer = 0
    ndim = len(ign)
    for i in range(0,64//ndim):  

        if ndim == 1:
            return ign[0]

        elif ndim == 2:
            bit_x = (ign[0] >> i) & 1
            bit_y = (ign[1] >> i) & 1

            answer |= (bit_x << (2*i)) | (bit_y << (2*i + 1))
        
        elif ndim == 3:
            bit_x = (ign[0] >> i) & 1
            bit_y = (ign[1] >> i) & 1
            bit_z = (ign[2] >> i) & 1

            answer |= (bit_x << (3*i)) | (bit_y << (3*i + 1)) | (bit_z << (3*i + 2))
        
    return answer

def inb(ign):
    x, y, z = ign
    def extract_bit(x):
        x = (x | (x << 16)) & 0x030000FF;
        x = (x | (x <<  8)) & 0x0300F00F;
        x = (x | (x <<  4)) & 0x030C30C3;
        x = (x | (x <<  2)) & 0x09249249;
        return x
    x, y, z= extract_bit(x), extract_bit(y), extract_bit(z)
    return x | (y << 1) | (z << 2)