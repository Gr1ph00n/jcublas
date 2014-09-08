/*
 * JCuda - Java bindings for NVIDIA CUDA jcuda.driver and jcuda.runtime API
 *
 * Copyright (c) 2009-2012 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

package jcuda.driver;

/**
 * Flags for mapping and unmapping interop resources
 */
public class CUgraphicsRegisterFlags
{
    public static final int CU_GRAPHICS_REGISTER_FLAGS_NONE  = 0x00;

    public static final int CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY     = 0x01;
    
    public static final int CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02;
    
    public static final int CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST  = 0x04;
    
    public static final int CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08;
    
    /**
     * Returns the String identifying the given CUgraphicsRegisterFlags
     *
     * @param n The CUgraphicsRegisterFlags
     * @return The String identifying the given CUgraphicsRegisterFlags
     */
    public static String stringFor(int n)
    {
        if (n == 0)
        {
            return "CU_GRAPHICS_REGISTER_FLAGS_NONE";
        }
        String result = "";
        if ((n & CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY    ) != 0) result += "CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY ";
        if ((n & CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) != 0) result += "CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD ";
        if ((n & CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST ) != 0) result += "CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST ";
        if ((n & CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER ) != 0) result += "CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER ";
        return result;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUgraphicsRegisterFlags()
    {
    }


}
