#include <miopen/miopen.h>
#include <gtest/gtest.h>

TEST(BackendApi, Tensor)
{
    miopenBackendDescriptor_t tensorDescriptor;

    miopenStatus_t status =
        miopenBackendCreateDescriptor(MIOPEN_BACKEND_TENSOR_DESCRIPTOR, &tensorDescriptor);
    ASSERT_TRUE(status == miopenStatusSuccess);

    status = miopenBackendFinalize(tensorDescriptor);
    EXPECT_FALSE(status == miopenStatusSuccess);

    int64_t theId = 1;
    status        = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_BOOLEAN, 1, &theId);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_INT64, 2, &theId);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_INT64, 1, &theId);
    EXPECT_TRUE(status == miopenStatusSuccess);

    miopenDataType_t theDataType = miopenFloat;
    status                       = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_CHAR, 1, &theDataType);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 2, &theDataType);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &theDataType);
    EXPECT_TRUE(status == miopenStatusSuccess);

    int64_t theAlignment = 8;
    status               = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT, MIOPEN_TYPE_CHAR, 1, &theAlignment);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT, MIOPEN_TYPE_INT64, 2, &theAlignment);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT, MIOPEN_TYPE_INT64, 1, &theAlignment);
    EXPECT_TRUE(status == miopenStatusSuccess);

    int64_t dimensions[5] = {4, 1, 16, 16, 0};
    status                = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT32, 4, &dimensions);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, 5, &dimensions);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, 4, &dimensions);
    EXPECT_TRUE(status == miopenStatusSuccess);

    int64_t strides[5] = {1 * 16 * 16, 16 * 16, 16, 1, 0};
    status             = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT32, 4, &strides);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT64, 5, &strides);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT64, 4, &strides);
    EXPECT_TRUE(status == miopenStatusSuccess);

    bool isVirtual = false;
    status         = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_CHAR, 1, &isVirtual);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_BOOLEAN, 2, &isVirtual);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_BOOLEAN, 1, &isVirtual);
    EXPECT_TRUE(status == miopenStatusSuccess);

    status = miopenBackendFinalize(tensorDescriptor);
    EXPECT_TRUE(status == miopenStatusSuccess);
}
