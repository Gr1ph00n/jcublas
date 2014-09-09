/*
 * JCusparse - Java bindings for CUSPARSE, the NVIDIA CUDA sparse
 * matrix library, to be used with JCuda
 *
 * Copyright (c) 2010-2012 Marco Hutter - http://www.jcuda.org
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

package jcuda.jcusparse;

import jcuda.*;
import jcuda.runtime.cudaStream_t;

/**
 * Java bindings for CUSPARSE, the NVIDIA CUDA sparse matrix
 * BLAS library.
 */
public class JCusparse
{
    /**
     * The flag that indicates whether the native library has been
     * loaded
     */
    private static boolean initialized = false;

    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not 
     * cusparseStatus.CUSPARSE_STATUS_SUCCESS
     */
    private static boolean exceptionsEnabled = false;
    
    /* Private constructor to prevent instantiation */
    private JCusparse()
    {
    }

    // Initialize the native library.
    static
    {
        initialize();
    }
    
    /**
     * Initializes the native library. Note that this method
     * does not have to be called explicitly, since it will
     * be called automatically when this class is loaded.
     */
    public static void initialize()
    {
        if (!initialized)
        {
            LibUtils.loadLibrary("JCusparse");
            initialized = true;
        }
    }
    

    /**
     * Set the specified log level for the JCusparse library.<br />
     * <br />
     * Currently supported log levels:
     * <br />
     * LOG_QUIET: Never print anything <br />
     * LOG_ERROR: Print error messages <br />
     * LOG_TRACE: Print a trace of all native function calls <br />
     *
     * @param logLevel The log level to use.
     */
    public static void setLogLevel(LogLevel logLevel)
    {
        setLogLevelNative(logLevel.ordinal());
    }

    private static native void setLogLevelNative(int logLevel);


    /**
     * Enables or disables exceptions. By default, the methods of this class
     * only set the {@link cusparseStatus} from the native methods. 
     * If exceptions are enabled, a CudaException with a detailed error 
     * message will be thrown if a method is about to set a result code 
     * that is not cusparseStatus.CUSPARSE_STATUS_SUCCESS
     * 
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }
    
    /**
     * If the given result is not cusparseStatus.CUSPARSE_STATUS_SUCCESS
     * and exceptions have been enabled, this method will throw a 
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     * 
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not cusparseStatus.CUSPARSE_STATUS_SUCCESS
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result != 
            cusparseStatus.CUSPARSE_STATUS_SUCCESS)
        {
            throw new CudaException(cusparseStatus.stringFor(result));
        }
        return result;
    }

    /**
     * If the given result is <strong>equal</strong> to 
     * cusparseStatus.JCUSPARSE_STATUS_INTERNAL_ERROR
     * and exceptions have been enabled, this method will throw a 
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.<br />
     * <br />
     * This method is used for the functions that do not return
     * an error code, but a constant value, like a cusparseFillMode.
     * The respective functions may still return internal errors
     * from the JNI part.
     * 
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is cusparseStatus.JCUSPARSE_STATUS_INTERNAL_ERROR
     */
    private static int checkForError(int result)
    {
        if (exceptionsEnabled && result == 
            cusparseStatus.JCUSPARSE_STATUS_INTERNAL_ERROR)
        {
            throw new CudaException(cusparseStatus.stringFor(result));
        }
        return result;
    }
    

    
    
    
    /** CUSPARSE initialization and managment routines */
    public static int cusparseCreate(
        cusparseHandle handle)
    {
        return checkResult(cusparseCreateNative(handle));
    }
    private static native int cusparseCreateNative(
        cusparseHandle handle);


    public static int cusparseDestroy(
        cusparseHandle handle)
    {
        return checkResult(cusparseDestroyNative(handle));
    }
    private static native int cusparseDestroyNative(
        cusparseHandle handle);


    public static int cusparseGetVersion(
        cusparseHandle handle, 
        int[] version)
    {
        return checkResult(cusparseGetVersionNative(handle, version));
    }
    private static native int cusparseGetVersionNative(
        cusparseHandle handle, 
        int[] version);


    public static int cusparseSetStream(
        cusparseHandle handle, 
        cudaStream_t streamId)
    {
        return checkResult(cusparseSetStreamNative(handle, streamId));
    }
    private static native int cusparseSetStreamNative(
        cusparseHandle handle, 
        cudaStream_t streamId);


    /** CUSPARSE type creation, destruction, set and get routines */
    public static int cusparseGetPointerMode(
        cusparseHandle handle, 
        int[] mode)
    {
        return checkResult(cusparseGetPointerModeNative(handle, mode));
    }
    private static native int cusparseGetPointerModeNative(
        cusparseHandle handle, 
        int[] mode);


    public static int cusparseSetPointerMode(
        cusparseHandle handle, 
        int mode)
    {
        return checkResult(cusparseSetPointerModeNative(handle, mode));
    }
    private static native int cusparseSetPointerModeNative(
        cusparseHandle handle, 
        int mode);


    public static int cusparseCreateMatDescr(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseCreateMatDescrNative(descrA));
    }
    private static native int cusparseCreateMatDescrNative(
        cusparseMatDescr descrA);


    public static int cusparseDestroyMatDescr(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseDestroyMatDescrNative(descrA));
    }
    private static native int cusparseDestroyMatDescrNative(
        cusparseMatDescr descrA);


    public static int cusparseSetMatType(
        cusparseMatDescr descrA, 
        int type)
    {
        return checkResult(cusparseSetMatTypeNative(descrA, type));
    }
    private static native int cusparseSetMatTypeNative(
        cusparseMatDescr descrA, 
        int type);


    public static int cusparseGetMatType(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseGetMatTypeNative(descrA));
    }
    private static native int cusparseGetMatTypeNative(
        cusparseMatDescr descrA);


    public static int cusparseSetMatFillMode(
        cusparseMatDescr descrA, 
        int fillMode)
    {
        return checkResult(cusparseSetMatFillModeNative(descrA, fillMode));
    }
    private static native int cusparseSetMatFillModeNative(
        cusparseMatDescr descrA, 
        int fillMode);


    public static int cusparseGetMatFillMode(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseGetMatFillModeNative(descrA));
    }
    private static native int cusparseGetMatFillModeNative(
        cusparseMatDescr descrA);


    public static int cusparseSetMatDiagType(
        cusparseMatDescr descrA, 
        int diagType)
    {
        return checkResult(cusparseSetMatDiagTypeNative(descrA, diagType));
    }
    private static native int cusparseSetMatDiagTypeNative(
        cusparseMatDescr descrA, 
        int diagType);


    public static int cusparseGetMatDiagType(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseGetMatDiagTypeNative(descrA));
    }
    private static native int cusparseGetMatDiagTypeNative(
        cusparseMatDescr descrA);


    public static int cusparseSetMatIndexBase(
        cusparseMatDescr descrA, 
        int base)
    {
        return checkResult(cusparseSetMatIndexBaseNative(descrA, base));
    }
    private static native int cusparseSetMatIndexBaseNative(
        cusparseMatDescr descrA, 
        int base);


    public static int cusparseGetMatIndexBase(
        cusparseMatDescr descrA)
    {
        return checkResult(cusparseGetMatIndexBaseNative(descrA));
    }
    private static native int cusparseGetMatIndexBaseNative(
        cusparseMatDescr descrA);


    /** sparse triangular solve */
    public static int cusparseCreateSolveAnalysisInfo(
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseCreateSolveAnalysisInfoNative(info));
    }
    private static native int cusparseCreateSolveAnalysisInfoNative(
        cusparseSolveAnalysisInfo info);


    public static int cusparseDestroySolveAnalysisInfo(
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseDestroySolveAnalysisInfoNative(info));
    }
    private static native int cusparseDestroySolveAnalysisInfoNative(
        cusparseSolveAnalysisInfo info);


    public static int cusparseGetLevelInfo(
        cusparseHandle handle, 
        cusparseSolveAnalysisInfo info, 
        int[] nlevels, 
        Pointer levelPtr, 
        Pointer levelInd)
    {
        return checkResult(cusparseGetLevelInfoNative(handle, info, nlevels, levelPtr, levelInd));
    }
    private static native int cusparseGetLevelInfoNative(
        cusparseHandle handle, 
        cusparseSolveAnalysisInfo info, 
        int[] nlevels, 
        Pointer levelPtr, 
        Pointer levelInd);


    public static int cusparseCreateCsrsv2Info(
        csrsv2Info info)
    {
        return checkResult(cusparseCreateCsrsv2InfoNative(info));
    }
    private static native int cusparseCreateCsrsv2InfoNative(
        csrsv2Info info);


    public static int cusparseDestroyCsrsv2Info(
        csrsv2Info info)
    {
        return checkResult(cusparseDestroyCsrsv2InfoNative(info));
    }
    private static native int cusparseDestroyCsrsv2InfoNative(
        csrsv2Info info);


    public static int cusparseCreateCsric02Info(
        csric02Info info)
    {
        return checkResult(cusparseCreateCsric02InfoNative(info));
    }
    private static native int cusparseCreateCsric02InfoNative(
        csric02Info info);


    public static int cusparseDestroyCsric02Info(
        csric02Info info)
    {
        return checkResult(cusparseDestroyCsric02InfoNative(info));
    }
    private static native int cusparseDestroyCsric02InfoNative(
        csric02Info info);


    public static int cusparseCreateBsric02Info(
        bsric02Info info)
    {
        return checkResult(cusparseCreateBsric02InfoNative(info));
    }
    private static native int cusparseCreateBsric02InfoNative(
        bsric02Info info);


    public static int cusparseDestroyBsric02Info(
        bsric02Info info)
    {
        return checkResult(cusparseDestroyBsric02InfoNative(info));
    }
    private static native int cusparseDestroyBsric02InfoNative(
        bsric02Info info);


    /** incomplete LU */
    public static int cusparseCreateCsrilu02Info(
        csrilu02Info info)
    {
        return checkResult(cusparseCreateCsrilu02InfoNative(info));
    }
    private static native int cusparseCreateCsrilu02InfoNative(
        csrilu02Info info);


    public static int cusparseDestroyCsrilu02Info(
        csrilu02Info info)
    {
        return checkResult(cusparseDestroyCsrilu02InfoNative(info));
    }
    private static native int cusparseDestroyCsrilu02InfoNative(
        csrilu02Info info);


    public static int cusparseCreateBsrilu02Info(
        bsrilu02Info info)
    {
        return checkResult(cusparseCreateBsrilu02InfoNative(info));
    }
    private static native int cusparseCreateBsrilu02InfoNative(
        bsrilu02Info info);


    public static int cusparseDestroyBsrilu02Info(
        bsrilu02Info info)
    {
        return checkResult(cusparseDestroyBsrilu02InfoNative(info));
    }
    private static native int cusparseDestroyBsrilu02InfoNative(
        bsrilu02Info info);


    /** BSR triangular solber */
    public static int cusparseCreateBsrsv2Info(
        bsrsv2Info info)
    {
        return checkResult(cusparseCreateBsrsv2InfoNative(info));
    }
    private static native int cusparseCreateBsrsv2InfoNative(
        bsrsv2Info info);


    public static int cusparseDestroyBsrsv2Info(
        bsrsv2Info info)
    {
        return checkResult(cusparseDestroyBsrsv2InfoNative(info));
    }
    private static native int cusparseDestroyBsrsv2InfoNative(
        bsrsv2Info info);


    public static int cusparseCreateBsrsm2Info(
        bsrsm2Info info)
    {
        return checkResult(cusparseCreateBsrsm2InfoNative(info));
    }
    private static native int cusparseCreateBsrsm2InfoNative(
        bsrsm2Info info);


    public static int cusparseDestroyBsrsm2Info(
        bsrsm2Info info)
    {
        return checkResult(cusparseDestroyBsrsm2InfoNative(info));
    }
    private static native int cusparseDestroyBsrsm2InfoNative(
        bsrsm2Info info);


    /** hybrid (HYB) format */
    public static int cusparseCreateHybMat(
        cusparseHybMat hybA)
    {
        return checkResult(cusparseCreateHybMatNative(hybA));
    }
    private static native int cusparseCreateHybMatNative(
        cusparseHybMat hybA);


    public static int cusparseDestroyHybMat(
        cusparseHybMat hybA)
    {
        return checkResult(cusparseDestroyHybMatNative(hybA));
    }
    private static native int cusparseDestroyHybMatNative(
        cusparseHybMat hybA);


    /** Description: Addition of a scalar multiple of a sparse vector x  
       and a dense vector y. */
    public static int cusparseSaxpyi(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseSaxpyiNative(handle, nnz, alpha, xVal, xInd, y, idxBase));
    }
    private static native int cusparseSaxpyiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    public static int cusparseDaxpyi(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseDaxpyiNative(handle, nnz, alpha, xVal, xInd, y, idxBase));
    }
    private static native int cusparseDaxpyiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    public static int cusparseCaxpyi(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseCaxpyiNative(handle, nnz, alpha, xVal, xInd, y, idxBase));
    }
    private static native int cusparseCaxpyiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    public static int cusparseZaxpyi(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseZaxpyiNative(handle, nnz, alpha, xVal, xInd, y, idxBase));
    }
    private static native int cusparseZaxpyiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer alpha, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    /** Description: dot product of a sparse vector x and a dense vector y. */
    public static int cusparseSdoti(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase)
    {
        return checkResult(cusparseSdotiNative(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase));
    }
    private static native int cusparseSdotiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase);


    public static int cusparseDdoti(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase)
    {
        return checkResult(cusparseDdotiNative(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase));
    }
    private static native int cusparseDdotiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase);


    public static int cusparseCdoti(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase)
    {
        return checkResult(cusparseCdotiNative(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase));
    }
    private static native int cusparseCdotiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase);


    public static int cusparseZdoti(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase)
    {
        return checkResult(cusparseZdotiNative(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase));
    }
    private static native int cusparseZdotiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase);


    /** Description: dot product of complex conjugate of a sparse vector x
       and a dense vector y. */
    public static int cusparseCdotci(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase)
    {
        return checkResult(cusparseCdotciNative(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase));
    }
    private static native int cusparseCdotciNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase);


    public static int cusparseZdotci(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase)
    {
        return checkResult(cusparseZdotciNative(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase));
    }
    private static native int cusparseZdotciNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer resultDevHostPtr, 
        int idxBase);


    /** Description: Gather of non-zero elements from dense vector y into 
       sparse vector x. */
    public static int cusparseSgthr(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseSgthrNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseSgthrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    public static int cusparseDgthr(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseDgthrNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseDgthrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    public static int cusparseCgthr(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseCgthrNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseCgthrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    public static int cusparseZgthr(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseZgthrNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseZgthrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    /** Description: Gather of non-zero elements from desne vector y into 
       sparse vector x (also replacing these elements in y by zeros). */
    public static int cusparseSgthrz(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseSgthrzNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseSgthrzNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    public static int cusparseDgthrz(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseDgthrzNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseDgthrzNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    public static int cusparseCgthrz(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseCgthrzNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseCgthrzNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    public static int cusparseZgthrz(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase)
    {
        return checkResult(cusparseZgthrzNative(handle, nnz, y, xVal, xInd, idxBase));
    }
    private static native int cusparseZgthrzNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer y, 
        Pointer xVal, 
        Pointer xInd, 
        int idxBase);


    /** Description: Scatter of elements of the sparse vector x into 
       dense vector y. */
    public static int cusparseSsctr(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseSsctrNative(handle, nnz, xVal, xInd, y, idxBase));
    }
    private static native int cusparseSsctrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    public static int cusparseDsctr(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseDsctrNative(handle, nnz, xVal, xInd, y, idxBase));
    }
    private static native int cusparseDsctrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    public static int cusparseCsctr(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseCsctrNative(handle, nnz, xVal, xInd, y, idxBase));
    }
    private static native int cusparseCsctrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    public static int cusparseZsctr(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase)
    {
        return checkResult(cusparseZsctrNative(handle, nnz, xVal, xInd, y, idxBase));
    }
    private static native int cusparseZsctrNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        int idxBase);


    /** Description: Givens rotation, where c and s are cosine and sine, 
       x and y are sparse and dense vectors, respectively. */
    public static int cusparseSroti(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer c, 
        Pointer s, 
        int idxBase)
    {
        return checkResult(cusparseSrotiNative(handle, nnz, xVal, xInd, y, c, s, idxBase));
    }
    private static native int cusparseSrotiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer c, 
        Pointer s, 
        int idxBase);


    public static int cusparseDroti(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer c, 
        Pointer s, 
        int idxBase)
    {
        return checkResult(cusparseDrotiNative(handle, nnz, xVal, xInd, y, c, s, idxBase));
    }
    private static native int cusparseDrotiNative(
        cusparseHandle handle, 
        int nnz, 
        Pointer xVal, 
        Pointer xInd, 
        Pointer y, 
        Pointer c, 
        Pointer s, 
        int idxBase);


    /** Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
       where A is a sparse matrix in CSR storage format, x and y are dense vectors. */
    public static int cusparseScsrmv(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseScsrmvNative(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y));
    }
    private static native int cusparseScsrmvNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseDcsrmv(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseDcsrmvNative(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y));
    }
    private static native int cusparseDcsrmvNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseCcsrmv(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseCcsrmvNative(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y));
    }
    private static native int cusparseCcsrmvNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseZcsrmv(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseZcsrmvNative(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y));
    }
    private static native int cusparseZcsrmvNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    /** Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
       where A is a sparse matrix in HYB storage format, x and y are dense vectors. */
    public static int cusparseShybmv(
        cusparseHandle handle, 
        int transA, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseShybmvNative(handle, transA, alpha, descrA, hybA, x, beta, y));
    }
    private static native int cusparseShybmvNative(
        cusparseHandle handle, 
        int transA, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseDhybmv(
        cusparseHandle handle, 
        int transA, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseDhybmvNative(handle, transA, alpha, descrA, hybA, x, beta, y));
    }
    private static native int cusparseDhybmvNative(
        cusparseHandle handle, 
        int transA, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseChybmv(
        cusparseHandle handle, 
        int transA, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseChybmvNative(handle, transA, alpha, descrA, hybA, x, beta, y));
    }
    private static native int cusparseChybmvNative(
        cusparseHandle handle, 
        int transA, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseZhybmv(
        cusparseHandle handle, 
        int transA, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseZhybmvNative(handle, transA, alpha, descrA, hybA, x, beta, y));
    }
    private static native int cusparseZhybmvNative(
        cusparseHandle handle, 
        int transA, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    /** Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
       where A is a sparse matrix in BSR storage format, x and y are dense vectors. */
    public static int cusparseSbsrmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseSbsrmvNative(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseSbsrmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseDbsrmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseDbsrmvNative(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseDbsrmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseCbsrmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseCbsrmvNative(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseCbsrmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseZbsrmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseZbsrmvNative(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseZbsrmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    /** Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
       where A is a sparse matrix in extended BSR storage format, x and y are dense 
       vectors. */
    public static int cusparseSbsrxmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrMaskPtrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrEndPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseSbsrxmvNative(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrValA, bsrMaskPtrA, bsrRowPtrA, bsrEndPtrA, bsrColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseSbsrxmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrMaskPtrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrEndPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseDbsrxmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrMaskPtrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrEndPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseDbsrxmvNative(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrValA, bsrMaskPtrA, bsrRowPtrA, bsrEndPtrA, bsrColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseDbsrxmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrMaskPtrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrEndPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseCbsrxmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrMaskPtrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrEndPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseCbsrxmvNative(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrValA, bsrMaskPtrA, bsrRowPtrA, bsrEndPtrA, bsrColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseCbsrxmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrMaskPtrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrEndPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    public static int cusparseZbsrxmv(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrMaskPtrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrEndPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y)
    {
        return checkResult(cusparseZbsrxmvNative(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrValA, bsrMaskPtrA, bsrRowPtrA, bsrEndPtrA, bsrColIndA, blockDim, x, beta, y));
    }
    private static native int cusparseZbsrxmvNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int sizeOfMask, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrMaskPtrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrEndPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        Pointer x, 
        Pointer beta, 
        Pointer y);


    /** Description: Solution of triangular linear system op(A) * y = alpha * x, 
       where A is a sparse matrix in CSR storage format, x and y are dense vectors. */
    public static int cusparseScsrsv_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseScsrsv_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseScsrsv_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseDcsrsv_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseDcsrsv_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseDcsrsv_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseCcsrsv_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseCcsrsv_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseCcsrsv_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseZcsrsv_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseZcsrsv_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseZcsrsv_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseScsrsv_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y)
    {
        return checkResult(cusparseScsrsv_solveNative(handle, transA, m, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, y));
    }
    private static native int cusparseScsrsv_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y);


    public static int cusparseDcsrsv_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y)
    {
        return checkResult(cusparseDcsrsv_solveNative(handle, transA, m, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, y));
    }
    private static native int cusparseDcsrsv_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y);


    public static int cusparseCcsrsv_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y)
    {
        return checkResult(cusparseCcsrsv_solveNative(handle, transA, m, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, y));
    }
    private static native int cusparseCcsrsv_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y);


    public static int cusparseZcsrsv_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y)
    {
        return checkResult(cusparseZcsrsv_solveNative(handle, transA, m, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, y));
    }
    private static native int cusparseZcsrsv_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y);


    /** Description: Solution of triangular linear system op(A) * y = alpha * x, 
       where A is a sparse matrix in CSR storage format, x and y are dense vectors. 
       The new API provides utility function to query size of buffer used. */
    public static int cusparseXcsrsv2_zeroPivot(
        cusparseHandle handle, 
        csrsv2Info info, 
        Pointer position)
    {
        return checkResult(cusparseXcsrsv2_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXcsrsv2_zeroPivotNative(
        cusparseHandle handle, 
        csrsv2Info info, 
        Pointer position);


    public static int cusparseScsrsv2_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseScsrsv2_bufferSizeNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseScsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseDcsrsv2_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsrsv2_bufferSizeNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseDcsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseCcsrsv2_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsrsv2_bufferSizeNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseCcsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseZcsrsv2_bufferSize(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsrsv2_bufferSizeNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseZcsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseScsrsv2_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrsv2_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseScsrsv2_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsrsv2_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrsv2_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseDcsrsv2_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsrsv2_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrsv2_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseCcsrsv2_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsrsv2_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrsv2_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseZcsrsv2_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseScsrsv2_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrsv2_solveNative(handle, transA, m, nnz, alpha, descra, csrValA, csrRowPtrA, csrColIndA, info, x, y, policy, pBuffer));
    }
    private static native int cusparseScsrsv2_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsrsv2_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrsv2_solveNative(handle, transA, m, nnz, alpha, descra, csrValA, csrRowPtrA, csrColIndA, info, x, y, policy, pBuffer));
    }
    private static native int cusparseDcsrsv2_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsrsv2_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrsv2_solveNative(handle, transA, m, nnz, alpha, descra, csrValA, csrRowPtrA, csrColIndA, info, x, y, policy, pBuffer));
    }
    private static native int cusparseCcsrsv2_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsrsv2_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrsv2_solveNative(handle, transA, m, nnz, alpha, descra, csrValA, csrRowPtrA, csrColIndA, info, x, y, policy, pBuffer));
    }
    private static native int cusparseZcsrsv2_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseXbsrsv2_zeroPivot(
        cusparseHandle handle, 
        bsrsv2Info info, 
        Pointer position)
    {
        return checkResult(cusparseXbsrsv2_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXbsrsv2_zeroPivotNative(
        cusparseHandle handle, 
        bsrsv2Info info, 
        Pointer position);


    public static int cusparseSbsrsv2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseSbsrsv2_bufferSizeNative(handle, dirA, transA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseSbsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseDbsrsv2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseDbsrsv2_bufferSizeNative(handle, dirA, transA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseDbsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseCbsrsv2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseCbsrsv2_bufferSizeNative(handle, dirA, transA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseCbsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseZbsrsv2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseZbsrsv2_bufferSizeNative(handle, dirA, transA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseZbsrsv2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseSbsrsv2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrsv2_analysisNative(handle, dirA, transA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseSbsrsv2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrsv2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrsv2_analysisNative(handle, dirA, transA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseDbsrsv2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrsv2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrsv2_analysisNative(handle, dirA, transA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseCbsrsv2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrsv2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrsv2_analysisNative(handle, dirA, transA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseZbsrsv2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseSbsrsv2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrsv2_solveNative(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, x, y, policy, pBuffer));
    }
    private static native int cusparseSbsrsv2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrsv2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrsv2_solveNative(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, x, y, policy, pBuffer));
    }
    private static native int cusparseDbsrsv2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrsv2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrsv2_solveNative(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, x, y, policy, pBuffer));
    }
    private static native int cusparseCbsrsv2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrsv2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrsv2_solveNative(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, x, y, policy, pBuffer));
    }
    private static native int cusparseZbsrsv2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int mb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrsv2Info info, 
        Pointer x, 
        Pointer y, 
        int policy, 
        Pointer pBuffer);


    /** Description: Solution of triangular linear system op(A) * y = alpha * x, 
       where A is a sparse matrix in HYB storage format, x and y are dense vectors. */
    public static int cusparseShybsv_analysis(
        cusparseHandle handle, 
        int transA, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseShybsv_analysisNative(handle, transA, descrA, hybA, info));
    }
    private static native int cusparseShybsv_analysisNative(
        cusparseHandle handle, 
        int transA, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseDhybsv_analysis(
        cusparseHandle handle, 
        int transA, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseDhybsv_analysisNative(handle, transA, descrA, hybA, info));
    }
    private static native int cusparseDhybsv_analysisNative(
        cusparseHandle handle, 
        int transA, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseChybsv_analysis(
        cusparseHandle handle, 
        int transA, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseChybsv_analysisNative(handle, transA, descrA, hybA, info));
    }
    private static native int cusparseChybsv_analysisNative(
        cusparseHandle handle, 
        int transA, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseZhybsv_analysis(
        cusparseHandle handle, 
        int transA, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseZhybsv_analysisNative(handle, transA, descrA, hybA, info));
    }
    private static native int cusparseZhybsv_analysisNative(
        cusparseHandle handle, 
        int transA, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseShybsv_solve(
        cusparseHandle handle, 
        int trans, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y)
    {
        return checkResult(cusparseShybsv_solveNative(handle, trans, alpha, descra, hybA, info, x, y));
    }
    private static native int cusparseShybsv_solveNative(
        cusparseHandle handle, 
        int trans, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y);


    public static int cusparseChybsv_solve(
        cusparseHandle handle, 
        int trans, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y)
    {
        return checkResult(cusparseChybsv_solveNative(handle, trans, alpha, descra, hybA, info, x, y));
    }
    private static native int cusparseChybsv_solveNative(
        cusparseHandle handle, 
        int trans, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y);


    public static int cusparseDhybsv_solve(
        cusparseHandle handle, 
        int trans, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y)
    {
        return checkResult(cusparseDhybsv_solveNative(handle, trans, alpha, descra, hybA, info, x, y));
    }
    private static native int cusparseDhybsv_solveNative(
        cusparseHandle handle, 
        int trans, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y);


    public static int cusparseZhybsv_solve(
        cusparseHandle handle, 
        int trans, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y)
    {
        return checkResult(cusparseZhybsv_solveNative(handle, trans, alpha, descra, hybA, info, x, y));
    }
    private static native int cusparseZhybsv_solveNative(
        cusparseHandle handle, 
        int trans, 
        Pointer alpha, 
        cusparseMatDescr descra, 
        cusparseHybMat hybA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        Pointer y);


    /** --- Sparse Level 3 routines --- */
    /** Description: Matrix-matrix multiplication C = alpha * op(A) * B  + beta * C, 
       where A is a sparse matrix, B and C are dense and usually tall matrices. */
    public static int cusparseScsrmm(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseScsrmmNative(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    }
    private static native int cusparseScsrmmNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseDcsrmm(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseDcsrmmNative(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    }
    private static native int cusparseDcsrmmNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseCcsrmm(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseCcsrmmNative(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    }
    private static native int cusparseCcsrmmNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseZcsrmm(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseZcsrmmNative(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    }
    private static native int cusparseZcsrmmNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseScsrmm2(
        cusparseHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseScsrmm2Native(handle, transa, transb, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    }
    private static native int cusparseScsrmm2Native(
        cusparseHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseDcsrmm2(
        cusparseHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseDcsrmm2Native(handle, transa, transb, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    }
    private static native int cusparseDcsrmm2Native(
        cusparseHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseCcsrmm2(
        cusparseHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseCcsrmm2Native(handle, transa, transb, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    }
    private static native int cusparseCcsrmm2Native(
        cusparseHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseZcsrmm2(
        cusparseHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseZcsrmm2Native(handle, transa, transb, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
    }
    private static native int cusparseZcsrmm2Native(
        cusparseHandle handle, 
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    /** Description: Solution of triangular linear system op(A) * Y = alpha * X, 
       with multiple right-hand-sides, where A is a sparse matrix in CSR storage 
       format, X and Y are dense and usually tall matrices. */
    public static int cusparseScsrsm_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseScsrsm_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseScsrsm_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseDcsrsm_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseDcsrsm_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseDcsrsm_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseCcsrsm_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseCcsrsm_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseCcsrsm_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseZcsrsm_analysis(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseZcsrsm_analysisNative(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseZcsrsm_analysisNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseScsrsm_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        int ldx, 
        Pointer y, 
        int ldy)
    {
        return checkResult(cusparseScsrsm_solveNative(handle, transA, m, n, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, ldx, y, ldy));
    }
    private static native int cusparseScsrsm_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        int ldx, 
        Pointer y, 
        int ldy);


    public static int cusparseDcsrsm_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        int ldx, 
        Pointer y, 
        int ldy)
    {
        return checkResult(cusparseDcsrsm_solveNative(handle, transA, m, n, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, ldx, y, ldy));
    }
    private static native int cusparseDcsrsm_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        int ldx, 
        Pointer y, 
        int ldy);


    public static int cusparseCcsrsm_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        int ldx, 
        Pointer y, 
        int ldy)
    {
        return checkResult(cusparseCcsrsm_solveNative(handle, transA, m, n, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, ldx, y, ldy));
    }
    private static native int cusparseCcsrsm_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        int ldx, 
        Pointer y, 
        int ldy);


    public static int cusparseZcsrsm_solve(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        int ldx, 
        Pointer y, 
        int ldy)
    {
        return checkResult(cusparseZcsrsm_solveNative(handle, transA, m, n, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, info, x, ldx, y, ldy));
    }
    private static native int cusparseZcsrsm_solveNative(
        cusparseHandle handle, 
        int transA, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info, 
        Pointer x, 
        int ldx, 
        Pointer y, 
        int ldy);


    public static int cusparseSbsrmm(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseSbsrmmNative(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockSize, B, ldb, beta, C, ldc));
    }
    private static native int cusparseSbsrmmNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseDbsrmm(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseDbsrmmNative(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockSize, B, ldb, beta, C, ldc));
    }
    private static native int cusparseDbsrmmNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseCbsrmm(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseCbsrmmNative(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockSize, B, ldb, beta, C, ldc));
    }
    private static native int cusparseCbsrmmNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseZbsrmm(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc)
    {
        return checkResult(cusparseZbsrmmNative(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockSize, B, ldb, beta, C, ldc));
    }
    private static native int cusparseZbsrmmNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transB, 
        int mb, 
        int n, 
        int kb, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockSize, 
        Pointer B, 
        int ldb, 
        Pointer beta, 
        Pointer C, 
        int ldc);


    public static int cusparseXbsrsm2_zeroPivot(
        cusparseHandle handle, 
        bsrsm2Info info, 
        Pointer position)
    {
        return checkResult(cusparseXbsrsm2_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXbsrsm2_zeroPivotNative(
        cusparseHandle handle, 
        bsrsm2Info info, 
        Pointer position);


    public static int cusparseSbsrsm2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseSbsrsm2_bufferSizeNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, pBufferSizeInBytes));
    }
    private static native int cusparseSbsrsm2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseDbsrsm2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseDbsrsm2_bufferSizeNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, pBufferSizeInBytes));
    }
    private static native int cusparseDbsrsm2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseCbsrsm2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseCbsrsm2_bufferSizeNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, pBufferSizeInBytes));
    }
    private static native int cusparseCbsrsm2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseZbsrsm2_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseZbsrsm2_bufferSizeNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, pBufferSizeInBytes));
    }
    private static native int cusparseZbsrsm2_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseSbsrsm2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrsm2_analysisNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, policy, pBuffer));
    }
    private static native int cusparseSbsrsm2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrsm2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrsm2_analysisNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, policy, pBuffer));
    }
    private static native int cusparseDbsrsm2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrsm2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrsm2_analysisNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, policy, pBuffer));
    }
    private static native int cusparseCbsrsm2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrsm2_analysis(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrsm2_analysisNative(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, policy, pBuffer));
    }
    private static native int cusparseZbsrsm2_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseSbsrsm2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer X, 
        int ldx, 
        Pointer Y, 
        int ldy, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrsm2_solveNative(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, X, ldx, Y, ldy, policy, pBuffer));
    }
    private static native int cusparseSbsrsm2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer X, 
        int ldx, 
        Pointer Y, 
        int ldy, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrsm2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer X, 
        int ldx, 
        Pointer Y, 
        int ldy, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrsm2_solveNative(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, X, ldx, Y, ldy, policy, pBuffer));
    }
    private static native int cusparseDbsrsm2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer X, 
        int ldx, 
        Pointer Y, 
        int ldy, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrsm2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer X, 
        int ldx, 
        Pointer Y, 
        int ldy, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrsm2_solveNative(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, X, ldx, Y, ldy, policy, pBuffer));
    }
    private static native int cusparseCbsrsm2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer X, 
        int ldx, 
        Pointer Y, 
        int ldy, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrsm2_solve(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer X, 
        int ldx, 
        Pointer Y, 
        int ldy, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrsm2_solveNative(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrVal, bsrRowPtr, bsrColInd, blockSize, info, X, ldx, Y, ldy, policy, pBuffer));
    }
    private static native int cusparseZbsrsm2_solveNative(
        cusparseHandle handle, 
        int dirA, 
        int transA, 
        int transXY, 
        int mb, 
        int n, 
        int nnzb, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockSize, 
        bsrsm2Info info, 
        Pointer X, 
        int ldx, 
        Pointer Y, 
        int ldy, 
        int policy, 
        Pointer pBuffer);


    /** --- Preconditioners --- */
    /** Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
       based on the information in the opaque structure info that was obtained 
       from the analysis phase (csrsv_analysis). */
    public static int cusparseScsrilu0(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseScsrilu0Native(handle, trans, m, descrA, csrValA_ValM, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseScsrilu0Native(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseDcsrilu0(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseDcsrilu0Native(handle, trans, m, descrA, csrValA_ValM, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseDcsrilu0Native(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseCcsrilu0(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseCcsrilu0Native(handle, trans, m, descrA, csrValA_ValM, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseCcsrilu0Native(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseZcsrilu0(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseZcsrilu0Native(handle, trans, m, descrA, csrValA_ValM, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseZcsrilu0Native(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseScsrilu02_numericBoost(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseScsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseScsrilu02_numericBoostNative(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseDcsrilu02_numericBoost(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseDcsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseDcsrilu02_numericBoostNative(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseCcsrilu02_numericBoost(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseCcsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseCcsrilu02_numericBoostNative(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseZcsrilu02_numericBoost(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseZcsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseZcsrilu02_numericBoostNative(
        cusparseHandle handle, 
        csrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseXcsrilu02_zeroPivot(
        cusparseHandle handle, 
        csrilu02Info info, 
        Pointer position)
    {
        return checkResult(cusparseXcsrilu02_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXcsrilu02_zeroPivotNative(
        cusparseHandle handle, 
        csrilu02Info info, 
        Pointer position);


    public static int cusparseScsrilu02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseScsrilu02_bufferSizeNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseScsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseDcsrilu02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsrilu02_bufferSizeNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseDcsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseCcsrilu02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsrilu02_bufferSizeNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseCcsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseZcsrilu02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsrilu02_bufferSizeNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseZcsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseScsrilu02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrilu02_analysisNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseScsrilu02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsrilu02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrilu02_analysisNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseDcsrilu02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsrilu02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrilu02_analysisNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseCcsrilu02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsrilu02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrilu02_analysisNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseZcsrilu02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseScsrilu02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                          to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsrilu02Native(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseScsrilu02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                          to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsrilu02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                          to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsrilu02Native(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseDcsrilu02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                          to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsrilu02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                          to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsrilu02Native(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseCcsrilu02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                          to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsrilu02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                          to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsrilu02Native(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseZcsrilu02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                          to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseSbsrilu02_numericBoost(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseSbsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseSbsrilu02_numericBoostNative(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseDbsrilu02_numericBoost(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseDbsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseDbsrilu02_numericBoostNative(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseCbsrilu02_numericBoost(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseCbsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseCbsrilu02_numericBoostNative(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseZbsrilu02_numericBoost(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val)
    {
        return checkResult(cusparseZbsrilu02_numericBoostNative(handle, info, enable_boost, tol, boost_val));
    }
    private static native int cusparseZbsrilu02_numericBoostNative(
        cusparseHandle handle, 
        bsrilu02Info info, 
        int enable_boost, 
        Pointer tol, 
        Pointer boost_val);


    public static int cusparseXbsrilu02_zeroPivot(
        cusparseHandle handle, 
        bsrilu02Info info, 
        Pointer position)
    {
        return checkResult(cusparseXbsrilu02_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXbsrilu02_zeroPivotNative(
        cusparseHandle handle, 
        bsrilu02Info info, 
        Pointer position);


    public static int cusparseSbsrilu02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseSbsrilu02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseSbsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseDbsrilu02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseDbsrilu02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseDbsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseCbsrilu02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseCbsrilu02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseCbsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseZbsrilu02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseZbsrilu02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseZbsrilu02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseSbsrilu02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrilu02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseSbsrilu02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrilu02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrilu02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseDbsrilu02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrilu02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrilu02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseCbsrilu02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrilu02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrilu02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseZbsrilu02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseSbsrilu02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descra, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsrilu02Native(handle, dirA, mb, nnzb, descra, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseSbsrilu02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descra, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsrilu02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descra, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsrilu02Native(handle, dirA, mb, nnzb, descra, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseDbsrilu02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descra, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsrilu02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descra, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsrilu02Native(handle, dirA, mb, nnzb, descra, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseCbsrilu02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descra, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsrilu02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descra, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsrilu02Native(handle, dirA, mb, nnzb, descra, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseZbsrilu02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descra, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsrilu02Info info, 
        int policy, 
        Pointer pBuffer);


    /** Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
       based on the information in the opaque structure info that was obtained 
       from the analysis phase (csrsv_analysis). */
    public static int cusparseScsric0(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseScsric0Native(handle, trans, m, descrA, csrValA_ValM, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseScsric0Native(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseDcsric0(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseDcsric0Native(handle, trans, m, descrA, csrValA_ValM, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseDcsric0Native(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseCcsric0(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseCcsric0Native(handle, trans, m, descrA, csrValA_ValM, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseCcsric0Native(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseZcsric0(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info)
    {
        return checkResult(cusparseZcsric0Native(handle, trans, m, descrA, csrValA_ValM, csrRowPtrA, csrColIndA, info));
    }
    private static native int cusparseZcsric0Native(
        cusparseHandle handle, 
        int trans, 
        int m, 
        cusparseMatDescr descrA, 
        Pointer csrValA_ValM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseSolveAnalysisInfo info);


    public static int cusparseXcsric02_zeroPivot(
        cusparseHandle handle, 
        csric02Info info, 
        Pointer position)
    {
        return checkResult(cusparseXcsric02_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXcsric02_zeroPivotNative(
        cusparseHandle handle, 
        csric02Info info, 
        Pointer position);


    public static int cusparseScsric02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseScsric02_bufferSizeNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseScsric02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseDcsric02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsric02_bufferSizeNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseDcsric02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseCcsric02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsric02_bufferSizeNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseCcsric02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseZcsric02_bufferSize(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsric02_bufferSizeNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes));
    }
    private static native int cusparseZcsric02_bufferSizeNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseScsric02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsric02_analysisNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseScsric02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsric02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsric02_analysisNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseDcsric02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsric02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsric02_analysisNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseCcsric02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsric02_analysis(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsric02_analysisNative(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseZcsric02_analysisNative(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseScsric02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsric02Native(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseScsric02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDcsric02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsric02Native(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseDcsric02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCcsric02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsric02Native(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseCcsric02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZcsric02(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsric02Native(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer));
    }
    private static native int cusparseZcsric02Native(
        cusparseHandle handle, 
        int m, 
        int nnz, 
        cusparseMatDescr descrA, 
        Pointer csrValA_valM, 
        /** matrix A values are updated inplace 
                                                         to be the preconditioner M values */
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        csric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseXbsric02_zeroPivot(
        cusparseHandle handle, 
        bsric02Info info, 
        Pointer position)
    {
        return checkResult(cusparseXbsric02_zeroPivotNative(handle, info, position));
    }
    private static native int cusparseXbsric02_zeroPivotNative(
        cusparseHandle handle, 
        bsric02Info info, 
        Pointer position);


    public static int cusparseSbsric02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseSbsric02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseSbsric02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseDbsric02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseDbsric02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseDbsric02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseCbsric02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseCbsric02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseCbsric02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseZbsric02_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseZbsric02_bufferSizeNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, pBufferSizeInBytes));
    }
    private static native int cusparseZbsric02_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        Pointer pBufferSizeInBytes);


    public static int cusparseSbsric02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer)
    {
        return checkResult(cusparseSbsric02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pInputBuffer));
    }
    private static native int cusparseSbsric02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer);


    public static int cusparseDbsric02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer)
    {
        return checkResult(cusparseDbsric02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pInputBuffer));
    }
    private static native int cusparseDbsric02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer);


    public static int cusparseCbsric02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer)
    {
        return checkResult(cusparseCbsric02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pInputBuffer));
    }
    private static native int cusparseCbsric02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer);


    public static int cusparseZbsric02_analysis(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer)
    {
        return checkResult(cusparseZbsric02_analysisNative(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pInputBuffer));
    }
    private static native int cusparseZbsric02_analysisNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pInputBuffer);


    public static int cusparseSbsric02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSbsric02Native(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseSbsric02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseDbsric02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDbsric02Native(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseDbsric02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseCbsric02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCbsric02Native(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseCbsric02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer);


    public static int cusparseZbsric02(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZbsric02Native(handle, dirA, mb, nnzb, descrA, bsrVal, bsrRowPtr, bsrColInd, blockDim, info, policy, pBuffer));
    }
    private static native int cusparseZbsric02Native(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int blockDim, 
        bsric02Info info, 
        int policy, 
        Pointer pBuffer);


    /**
     * <pre>
     * Description: Solution of tridiagonal linear system A * B = B,
       with multiple right-hand-sides. The coefficient matrix A is 
       composed of lower (dl), main (d) and upper (du) diagonals, and 
       the right-hand-sides B are overwritten with the solution. 
     * These routine use pivoting
     * </pre>
     */
    public static int cusparseSgtsv(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cusparseSgtsvNative(handle, m, n, dl, d, du, B, ldb));
    }
    private static native int cusparseSgtsvNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb);


    public static int cusparseDgtsv(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cusparseDgtsvNative(handle, m, n, dl, d, du, B, ldb));
    }
    private static native int cusparseDgtsvNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb);


    public static int cusparseCgtsv(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cusparseCgtsvNative(handle, m, n, dl, d, du, B, ldb));
    }
    private static native int cusparseCgtsvNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb);


    public static int cusparseZgtsv(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cusparseZgtsvNative(handle, m, n, dl, d, du, B, ldb));
    }
    private static native int cusparseZgtsvNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb);


    /**
     * <pre>
     * Description: Solution of tridiagonal linear system A * B = B,
       with multiple right-hand-sides. The coefficient matrix A is 
       composed of lower (dl), main (d) and upper (du) diagonals, and 
       the right-hand-sides B are overwritten with the solution. 
     * These routines do not use pivoting, using a combination of PCR and CR algorithm
     * </pre>
     */
    public static int cusparseSgtsv_nopivot(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cusparseSgtsv_nopivotNative(handle, m, n, dl, d, du, B, ldb));
    }
    private static native int cusparseSgtsv_nopivotNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb);


    public static int cusparseDgtsv_nopivot(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cusparseDgtsv_nopivotNative(handle, m, n, dl, d, du, B, ldb));
    }
    private static native int cusparseDgtsv_nopivotNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb);


    public static int cusparseCgtsv_nopivot(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cusparseCgtsv_nopivotNative(handle, m, n, dl, d, du, B, ldb));
    }
    private static native int cusparseCgtsv_nopivotNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb);


    public static int cusparseZgtsv_nopivot(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb)
    {
        return checkResult(cusparseZgtsv_nopivotNative(handle, m, n, dl, d, du, B, ldb));
    }
    private static native int cusparseZgtsv_nopivotNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer B, 
        int ldb);


    /**
     * <pre>
     * Description: Solution of a set of tridiagonal linear systems
       A * x = x, each with a single right-hand-side. The coefficient 
       matrices A are composed of lower (dl), main (d) and upper (du) 
       diagonals and stored separated by a batchStride, while the 
     * right-hand-sides x are also separated by a batchStride.
     * </pre>
     */
    public static int cusparseSgtsvStridedBatch(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride)
    {
        return checkResult(cusparseSgtsvStridedBatchNative(handle, m, dl, d, du, x, batchCount, batchStride));
    }
    private static native int cusparseSgtsvStridedBatchNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride);


    public static int cusparseDgtsvStridedBatch(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride)
    {
        return checkResult(cusparseDgtsvStridedBatchNative(handle, m, dl, d, du, x, batchCount, batchStride));
    }
    private static native int cusparseDgtsvStridedBatchNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride);


    public static int cusparseCgtsvStridedBatch(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride)
    {
        return checkResult(cusparseCgtsvStridedBatchNative(handle, m, dl, d, du, x, batchCount, batchStride));
    }
    private static native int cusparseCgtsvStridedBatchNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride);


    public static int cusparseZgtsvStridedBatch(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride)
    {
        return checkResult(cusparseZgtsvStridedBatchNative(handle, m, dl, d, du, x, batchCount, batchStride));
    }
    private static native int cusparseZgtsvStridedBatchNative(
        cusparseHandle handle, 
        int m, 
        Pointer dl, 
        Pointer d, 
        Pointer du, 
        Pointer x, 
        int batchCount, 
        int batchStride);


    /** --- Extra --- */
    /** Description: This routine computes a sparse matrix that results from 
       multiplication of two sparse matrices. */
    public static int cusparseXcsrgemmNnz(
        cusparseHandle handle, 
        int transA, 
        int transB, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseXcsrgemmNnzNative(handle, transA, transB, m, n, k, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC, nnzTotalDevHostPtr));
    }
    private static native int cusparseXcsrgemmNnzNative(
        cusparseHandle handle, 
        int transA, 
        int transB, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer nnzTotalDevHostPtr);


    public static int cusparseScsrgemm(
        cusparseHandle handle, 
        int transA, 
        int transB, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseScsrgemmNative(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseScsrgemmNative(
        cusparseHandle handle, 
        int transA, 
        int transB, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseDcsrgemm(
        cusparseHandle handle, 
        int transA, 
        int transB, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseDcsrgemmNative(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseDcsrgemmNative(
        cusparseHandle handle, 
        int transA, 
        int transB, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseCcsrgemm(
        cusparseHandle handle, 
        int transA, 
        int transB, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseCcsrgemmNative(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseCcsrgemmNative(
        cusparseHandle handle, 
        int transA, 
        int transB, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseZcsrgemm(
        cusparseHandle handle, 
        int transA, 
        int transB, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseZcsrgemmNative(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseZcsrgemmNative(
        cusparseHandle handle, 
        int transA, 
        int transB, 
        int m, 
        int n, 
        int k, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    /** Description: This routine computes a sparse matrix that results from 
       addition of two sparse matrices. */
    public static int cusparseXcsrgeamNnz(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseXcsrgeamNnzNative(handle, m, n, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC, nnzTotalDevHostPtr));
    }
    private static native int cusparseXcsrgeamNnzNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer nnzTotalDevHostPtr);


    public static int cusparseScsrgeam(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseScsrgeamNative(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseScsrgeamNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseDcsrgeam(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseDcsrgeamNative(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseDcsrgeamNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseCcsrgeam(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseCcsrgeamNative(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseCcsrgeamNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseZcsrgeam(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseZcsrgeamNative(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseZcsrgeamNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        Pointer alpha, 
        cusparseMatDescr descrA, 
        int nnzA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer beta, 
        cusparseMatDescr descrB, 
        int nnzB, 
        Pointer csrValB, 
        Pointer csrRowPtrB, 
        Pointer csrColIndB, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    /** --- Sparse Format Conversion --- */
    /** Description: This routine finds the total number of non-zero elements and 
       the number of non-zero elements per row or column in the dense matrix A. */
    public static int cusparseSnnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseSnnzNative(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr));
    }
    private static native int cusparseSnnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr);


    public static int cusparseDnnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseDnnzNative(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr));
    }
    private static native int cusparseDnnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr);


    public static int cusparseCnnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseCnnzNative(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr));
    }
    private static native int cusparseCnnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr);


    public static int cusparseZnnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseZnnzNative(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr));
    }
    private static native int cusparseZnnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRowCol, 
        Pointer nnzTotalDevHostPtr);


    /** Description: This routine converts a dense matrix to a sparse matrix 
       in the CSR storage format, using the information computed by the 
       nnz routine. */
    public static int cusparseSdense2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA)
    {
        return checkResult(cusparseSdense2csrNative(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA));
    }
    private static native int cusparseSdense2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA);


    public static int cusparseDdense2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA)
    {
        return checkResult(cusparseDdense2csrNative(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA));
    }
    private static native int cusparseDdense2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA);


    public static int cusparseCdense2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA)
    {
        return checkResult(cusparseCdense2csrNative(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA));
    }
    private static native int cusparseCdense2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA);


    public static int cusparseZdense2csr(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA)
    {
        return checkResult(cusparseZdense2csrNative(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA));
    }
    private static native int cusparseZdense2csrNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA);


    /** Description: This routine converts a sparse matrix in CSR storage format
       to a dense matrix. */
    public static int cusparseScsr2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseScsr2denseNative(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda));
    }
    private static native int cusparseScsr2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer A, 
        int lda);


    public static int cusparseDcsr2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseDcsr2denseNative(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda));
    }
    private static native int cusparseDcsr2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer A, 
        int lda);


    public static int cusparseCcsr2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseCcsr2denseNative(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda));
    }
    private static native int cusparseCcsr2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer A, 
        int lda);


    public static int cusparseZcsr2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseZcsr2denseNative(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda));
    }
    private static native int cusparseZcsr2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        Pointer A, 
        int lda);


    /** Description: This routine converts a dense matrix to a sparse matrix 
       in the CSC storage format, using the information computed by the 
       nnz routine. */
    public static int cusparseSdense2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA)
    {
        return checkResult(cusparseSdense2cscNative(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA));
    }
    private static native int cusparseSdense2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA);


    public static int cusparseDdense2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA)
    {
        return checkResult(cusparseDdense2cscNative(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA));
    }
    private static native int cusparseDdense2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA);


    public static int cusparseCdense2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA)
    {
        return checkResult(cusparseCdense2cscNative(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA));
    }
    private static native int cusparseCdense2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA);


    public static int cusparseZdense2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA)
    {
        return checkResult(cusparseZdense2cscNative(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA));
    }
    private static native int cusparseZdense2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerCol, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA);


    /** Description: This routine converts a sparse matrix in CSC storage format
       to a dense matrix. */
    public static int cusparseScsc2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseScsc2denseNative(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda));
    }
    private static native int cusparseScsc2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        Pointer A, 
        int lda);


    public static int cusparseDcsc2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseDcsc2denseNative(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda));
    }
    private static native int cusparseDcsc2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        Pointer A, 
        int lda);


    public static int cusparseCcsc2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseCcsc2denseNative(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda));
    }
    private static native int cusparseCcsc2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        Pointer A, 
        int lda);


    public static int cusparseZcsc2dense(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseZcsc2denseNative(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda));
    }
    private static native int cusparseZcsc2denseNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        Pointer A, 
        int lda);


    /** Description: This routine compresses the indecis of rows or columns.
       It can be interpreted as a conversion from COO to CSR sparse storage
       format. */
    public static int cusparseXcoo2csr(
        cusparseHandle handle, 
        Pointer cooRowInd, 
        int nnz, 
        int m, 
        Pointer csrRowPtr, 
        int idxBase)
    {
        return checkResult(cusparseXcoo2csrNative(handle, cooRowInd, nnz, m, csrRowPtr, idxBase));
    }
    private static native int cusparseXcoo2csrNative(
        cusparseHandle handle, 
        Pointer cooRowInd, 
        int nnz, 
        int m, 
        Pointer csrRowPtr, 
        int idxBase);


    /** Description: This routine uncompresses the indecis of rows or columns.
       It can be interpreted as a conversion from CSR to COO sparse storage
       format. */
    public static int cusparseXcsr2coo(
        cusparseHandle handle, 
        Pointer csrRowPtr, 
        int nnz, 
        int m, 
        Pointer cooRowInd, 
        int idxBase)
    {
        return checkResult(cusparseXcsr2cooNative(handle, csrRowPtr, nnz, m, cooRowInd, idxBase));
    }
    private static native int cusparseXcsr2cooNative(
        cusparseHandle handle, 
        Pointer csrRowPtr, 
        int nnz, 
        int m, 
        Pointer cooRowInd, 
        int idxBase);


    /** Description: This routine converts a matrix from CSR to CSC sparse 
       storage format. The resulting matrix can be re-interpreted as a 
       transpose of the original matrix in CSR storage format. */
    public static int cusparseScsr2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr, 
        int copyValues, 
        int idxBase)
    {
        return checkResult(cusparseScsr2cscNative(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase));
    }
    private static native int cusparseScsr2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr, 
        int copyValues, 
        int idxBase);


    public static int cusparseDcsr2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr, 
        int copyValues, 
        int idxBase)
    {
        return checkResult(cusparseDcsr2cscNative(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase));
    }
    private static native int cusparseDcsr2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr, 
        int copyValues, 
        int idxBase);


    public static int cusparseCcsr2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr, 
        int copyValues, 
        int idxBase)
    {
        return checkResult(cusparseCcsr2cscNative(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase));
    }
    private static native int cusparseCcsr2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr, 
        int copyValues, 
        int idxBase);


    public static int cusparseZcsr2csc(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr, 
        int copyValues, 
        int idxBase)
    {
        return checkResult(cusparseZcsr2cscNative(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase));
    }
    private static native int cusparseZcsr2cscNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        int nnz, 
        Pointer csrVal, 
        Pointer csrRowPtr, 
        Pointer csrColInd, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr, 
        int copyValues, 
        int idxBase);


    /** Description: This routine converts a dense matrix to a sparse matrix 
       in HYB storage format. */
    public static int cusparseSdense2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseSdense2hybNative(handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseSdense2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    public static int cusparseDdense2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseDdense2hybNative(handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseDdense2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    public static int cusparseCdense2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseCdense2hybNative(handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseCdense2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    public static int cusparseZdense2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseZdense2hybNative(handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseZdense2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer A, 
        int lda, 
        Pointer nnzPerRow, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    /** Description: This routine converts a sparse matrix in HYB storage format
       to a dense matrix. */
    public static int cusparseShyb2dense(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseShyb2denseNative(handle, descrA, hybA, A, lda));
    }
    private static native int cusparseShyb2denseNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer A, 
        int lda);


    public static int cusparseDhyb2dense(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseDhyb2denseNative(handle, descrA, hybA, A, lda));
    }
    private static native int cusparseDhyb2denseNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer A, 
        int lda);


    public static int cusparseChyb2dense(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseChyb2denseNative(handle, descrA, hybA, A, lda));
    }
    private static native int cusparseChyb2denseNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer A, 
        int lda);


    public static int cusparseZhyb2dense(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer A, 
        int lda)
    {
        return checkResult(cusparseZhyb2denseNative(handle, descrA, hybA, A, lda));
    }
    private static native int cusparseZhyb2denseNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer A, 
        int lda);


    /** Description: This routine converts a sparse matrix in CSR storage format
       to a sparse matrix in HYB storage format. */
    public static int cusparseScsr2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseScsr2hybNative(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseScsr2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    public static int cusparseDcsr2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseDcsr2hybNative(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseDcsr2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    public static int cusparseCcsr2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseCcsr2hybNative(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseCcsr2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    public static int cusparseZcsr2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseZcsr2hybNative(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseZcsr2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    /** Description: This routine converts a sparse matrix in HYB storage format
       to a sparse matrix in CSR storage format. */
    public static int cusparseShyb2csr(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA)
    {
        return checkResult(cusparseShyb2csrNative(handle, descrA, hybA, csrValA, csrRowPtrA, csrColIndA));
    }
    private static native int cusparseShyb2csrNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA);


    public static int cusparseDhyb2csr(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA)
    {
        return checkResult(cusparseDhyb2csrNative(handle, descrA, hybA, csrValA, csrRowPtrA, csrColIndA));
    }
    private static native int cusparseDhyb2csrNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA);


    public static int cusparseChyb2csr(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA)
    {
        return checkResult(cusparseChyb2csrNative(handle, descrA, hybA, csrValA, csrRowPtrA, csrColIndA));
    }
    private static native int cusparseChyb2csrNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA);


    public static int cusparseZhyb2csr(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA)
    {
        return checkResult(cusparseZhyb2csrNative(handle, descrA, hybA, csrValA, csrRowPtrA, csrColIndA));
    }
    private static native int cusparseZhyb2csrNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA);


    /** Description: This routine converts a sparse matrix in CSC storage format
       to a sparse matrix in HYB storage format. */
    public static int cusparseScsc2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseScsc2hybNative(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseScsc2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    public static int cusparseDcsc2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseDcsc2hybNative(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseDcsc2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    public static int cusparseCcsc2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseCcsc2hybNative(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseCcsc2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    public static int cusparseZcsc2hyb(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType)
    {
        return checkResult(cusparseZcsc2hybNative(handle, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, hybA, userEllWidth, partitionType));
    }
    private static native int cusparseZcsc2hybNative(
        cusparseHandle handle, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer cscValA, 
        Pointer cscRowIndA, 
        Pointer cscColPtrA, 
        cusparseHybMat hybA, 
        int userEllWidth, 
        int partitionType);


    /** Description: This routine converts a sparse matrix in HYB storage format
       to a sparse matrix in CSC storage format. */
    public static int cusparseShyb2csc(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr)
    {
        return checkResult(cusparseShyb2cscNative(handle, descrA, hybA, cscVal, cscRowInd, cscColPtr));
    }
    private static native int cusparseShyb2cscNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr);


    public static int cusparseDhyb2csc(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr)
    {
        return checkResult(cusparseDhyb2cscNative(handle, descrA, hybA, cscVal, cscRowInd, cscColPtr));
    }
    private static native int cusparseDhyb2cscNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr);


    public static int cusparseChyb2csc(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr)
    {
        return checkResult(cusparseChyb2cscNative(handle, descrA, hybA, cscVal, cscRowInd, cscColPtr));
    }
    private static native int cusparseChyb2cscNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr);


    public static int cusparseZhyb2csc(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr)
    {
        return checkResult(cusparseZhyb2cscNative(handle, descrA, hybA, cscVal, cscRowInd, cscColPtr));
    }
    private static native int cusparseZhyb2cscNative(
        cusparseHandle handle, 
        cusparseMatDescr descrA, 
        cusparseHybMat hybA, 
        Pointer cscVal, 
        Pointer cscRowInd, 
        Pointer cscColPtr);


    /** Description: This routine converts a sparse matrix in CSR storage format
       to a sparse matrix in BSR storage format. */
    public static int cusparseXcsr2bsrNnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrRowPtrC, 
        Pointer nnzTotalDevHostPtr)
    {
        return checkResult(cusparseXcsr2bsrNnzNative(handle, dirA, m, n, descrA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrRowPtrC, nnzTotalDevHostPtr));
    }
    private static native int cusparseXcsr2bsrNnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrRowPtrC, 
        Pointer nnzTotalDevHostPtr);


    public static int cusparseScsr2bsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC)
    {
        return checkResult(cusparseScsr2bsrNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC));
    }
    private static native int cusparseScsr2bsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC);


    public static int cusparseDcsr2bsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC)
    {
        return checkResult(cusparseDcsr2bsrNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC));
    }
    private static native int cusparseDcsr2bsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC);


    public static int cusparseCcsr2bsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC)
    {
        return checkResult(cusparseCcsr2bsrNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC));
    }
    private static native int cusparseCcsr2bsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC);


    public static int cusparseZcsr2bsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC)
    {
        return checkResult(cusparseZcsr2bsrNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC));
    }
    private static native int cusparseZcsr2bsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC);


    /** Description: This routine converts a sparse matrix in BSR storage format
       to a sparse matrix in CSR storage format. */
    public static int cusparseSbsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseSbsr2csrNative(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseSbsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseDbsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseDbsr2csrNative(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseDbsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseCbsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseCbsr2csrNative(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseCbsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseZbsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseZbsr2csrNative(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseZbsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int blockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseSgebsr2gebsc_bufferSize(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseSgebsr2gebsc_bufferSizeNative(handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseSgebsr2gebsc_bufferSizeNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes);


    public static int cusparseDgebsr2gebsc_bufferSize(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseDgebsr2gebsc_bufferSizeNative(handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseDgebsr2gebsc_bufferSizeNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes);


    public static int cusparseCgebsr2gebsc_bufferSize(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseCgebsr2gebsc_bufferSizeNative(handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseCgebsr2gebsc_bufferSizeNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes);


    public static int cusparseZgebsr2gebsc_bufferSize(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseZgebsr2gebsc_bufferSizeNative(handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseZgebsr2gebsc_bufferSizeNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes);


    public static int cusparseSgebsr2gebsc(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int baseIdx, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSgebsr2gebscNative(handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, baseIdx, pBuffer));
    }
    private static native int cusparseSgebsr2gebscNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int baseIdx, 
        Pointer pBuffer);


    public static int cusparseDgebsr2gebsc(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int baseIdx, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDgebsr2gebscNative(handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, baseIdx, pBuffer));
    }
    private static native int cusparseDgebsr2gebscNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int baseIdx, 
        Pointer pBuffer);


    public static int cusparseCgebsr2gebsc(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int baseIdx, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCgebsr2gebscNative(handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, baseIdx, pBuffer));
    }
    private static native int cusparseCgebsr2gebscNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int baseIdx, 
        Pointer pBuffer);


    public static int cusparseZgebsr2gebsc(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int baseIdx, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZgebsr2gebscNative(handle, mb, nb, nnzb, bsrVal, bsrRowPtr, bsrColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, baseIdx, pBuffer));
    }
    private static native int cusparseZgebsr2gebscNative(
        cusparseHandle handle, 
        int mb, 
        int nb, 
        int nnzb, 
        Pointer bsrVal, 
        Pointer bsrRowPtr, 
        Pointer bsrColInd, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer bscVal, 
        Pointer bscRowInd, 
        Pointer bscColPtr, 
        int copyValues, 
        int baseIdx, 
        Pointer pBuffer);


    public static int cusparseXgebsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseXgebsr2csrNative(handle, dirA, mb, nb, descrA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, descrC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseXgebsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseSgebsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseSgebsr2csrNative(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseSgebsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseDgebsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseDgebsr2csrNative(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseDgebsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseCgebsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseCgebsr2csrNative(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseCgebsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseZgebsr2csr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC)
    {
        return checkResult(cusparseZgebsr2csrNative(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDim, colBlockDim, descrC, csrValC, csrRowPtrC, csrColIndC));
    }
    private static native int cusparseZgebsr2csrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        cusparseMatDescr descrC, 
        Pointer csrValC, 
        Pointer csrRowPtrC, 
        Pointer csrColIndC);


    public static int cusparseScsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseScsr2gebsr_bufferSizeNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseScsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes);


    public static int cusparseDcsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseDcsr2gebsr_bufferSizeNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseDcsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes);


    public static int cusparseCcsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseCcsr2gebsr_bufferSizeNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseCcsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes);


    public static int cusparseZcsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseZcsr2gebsr_bufferSizeNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes));
    }
    private static native int cusparseZcsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBufferSizeInBytes);


    public static int cusparseXcsr2gebsrNnz(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrRowPtrC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer)
    {
        return checkResult(cusparseXcsr2gebsrNnzNative(handle, dirA, m, n, descrA, csrRowPtrA, csrColIndA, descrC, bsrRowPtrC, rowBlockDim, colBlockDim, nnzTotalDevHostPtr, pBuffer));
    }
    private static native int cusparseXcsr2gebsrNnzNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrRowPtrC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer);


    public static int cusparseScsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer)
    {
        return checkResult(cusparseScsr2gebsrNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDim, colBlockDim, pBuffer));
    }
    private static native int cusparseScsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer);


    public static int cusparseDcsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDcsr2gebsrNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDim, colBlockDim, pBuffer));
    }
    private static native int cusparseDcsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer);


    public static int cusparseCcsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCcsr2gebsrNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDim, colBlockDim, pBuffer));
    }
    private static native int cusparseCcsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer);


    public static int cusparseZcsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZcsr2gebsrNative(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDim, colBlockDim, pBuffer));
    }
    private static native int cusparseZcsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int m, 
        int n, 
        cusparseMatDescr descrA, 
        Pointer csrValA, 
        Pointer csrRowPtrA, 
        Pointer csrColIndA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDim, 
        int colBlockDim, 
        Pointer pBuffer);


    public static int cusparseSgebsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseSgebsr2gebsr_bufferSizeNative(handle, dirA, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes));
    }
    private static native int cusparseSgebsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBufferSizeInBytes);


    public static int cusparseDgebsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseDgebsr2gebsr_bufferSizeNative(handle, dirA, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes));
    }
    private static native int cusparseDgebsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBufferSizeInBytes);


    public static int cusparseCgebsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseCgebsr2gebsr_bufferSizeNative(handle, dirA, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes));
    }
    private static native int cusparseCgebsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBufferSizeInBytes);


    public static int cusparseZgebsr2gebsr_bufferSize(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBufferSizeInBytes)
    {
        return checkResult(cusparseZgebsr2gebsr_bufferSizeNative(handle, dirA, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes));
    }
    private static native int cusparseZgebsr2gebsr_bufferSizeNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBufferSizeInBytes);


    public static int cusparseXgebsr2gebsrNnz(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrRowPtrC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer)
    {
        return checkResult(cusparseXgebsr2gebsrNnzNative(handle, dirA, mb, nb, nnzb, descrA, bsrRowPtrA, bsrColIndA, rowBlockDimA, colBlockDimA, descrC, bsrRowPtrC, rowBlockDimC, colBlockDimC, nnzTotalDevHostPtr, pBuffer));
    }
    private static native int cusparseXgebsr2gebsrNnzNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrRowPtrC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer nnzTotalDevHostPtr, 
        Pointer pBuffer);


    public static int cusparseSgebsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseSgebsr2gebsrNative(handle, dirA, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA, colBlockDimA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDimC, colBlockDimC, pBuffer));
    }
    private static native int cusparseSgebsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer);


    public static int cusparseDgebsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseDgebsr2gebsrNative(handle, dirA, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA, colBlockDimA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDimC, colBlockDimC, pBuffer));
    }
    private static native int cusparseDgebsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer);


    public static int cusparseCgebsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseCgebsr2gebsrNative(handle, dirA, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA, colBlockDimA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDimC, colBlockDimC, pBuffer));
    }
    private static native int cusparseCgebsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer);


    public static int cusparseZgebsr2gebsr(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer)
    {
        return checkResult(cusparseZgebsr2gebsrNative(handle, dirA, mb, nb, nnzb, descrA, bsrValA, bsrRowPtrA, bsrColIndA, rowBlockDimA, colBlockDimA, descrC, bsrValC, bsrRowPtrC, bsrColIndC, rowBlockDimC, colBlockDimC, pBuffer));
    }
    private static native int cusparseZgebsr2gebsrNative(
        cusparseHandle handle, 
        int dirA, 
        int mb, 
        int nb, 
        int nnzb, 
        cusparseMatDescr descrA, 
        Pointer bsrValA, 
        Pointer bsrRowPtrA, 
        Pointer bsrColIndA, 
        int rowBlockDimA, 
        int colBlockDimA, 
        cusparseMatDescr descrC, 
        Pointer bsrValC, 
        Pointer bsrRowPtrC, 
        Pointer bsrColIndC, 
        int rowBlockDimC, 
        int colBlockDimC, 
        Pointer pBuffer);


}
