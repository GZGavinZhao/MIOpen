#pragma once

#include <miopen/miopen.h>
#include <vector>

namespace miopen {

namespace backend_api {

class BackendDescriptor
{
public:
    BackendDescriptor() : m_IsFinalized(false) {}
    virtual ~BackendDescriptor()                               = default;
    virtual miopenStatus_t SetAttribute(miopenBackendAttributeName_t attributeName,
                                        miopenBackendAttributeType_t attributeType,
                                        int64_t elementCount,
                                        void* arrayOfElements) = 0;
    virtual miopenStatus_t Finalize()                          = 0;

protected:
    bool m_IsFinalized;
};

class BackendTensorDescriptor : public BackendDescriptor
{
public:
    BackendTensorDescriptor();
    virtual ~BackendTensorDescriptor() override;
    virtual miopenStatus_t SetAttribute(miopenBackendAttributeName_t attributeName,
                                        miopenBackendAttributeType_t attributeType,
                                        int64_t elementCount,
                                        void* arrayOfElements) override;
    virtual miopenStatus_t Finalize() override;
    inline miopenStatus_t Status() const noexcept { return m_Status; }
    inline bool IsVirtual() const noexcept { return m_IsVirtual; }

private:
    miopenTensorDescriptor_t m_Descriptor;
    std::vector<int64_t> m_Dimensions;
    std::vector<int64_t> m_Strides;
    int64_t m_UniqueId;
    int64_t m_ByteAlignment;
    int64_t m_VectorCount;
    int64_t m_VectorizedDimension;
    miopenBackendDescriptor_t m_RaggedOffsetDesc;
    miopenDataType_t m_DataType;
    miopenStatus_t m_Status;
    bool m_IsVirtual;
    bool m_UniqueIdSet;
    bool m_DataTypeSet;
    bool m_ByteAlignmentSet;
    bool m_DimensionsSet;
    bool m_StridesSet;
    bool m_VectorCountSet;
    bool m_VectorizedDimensionSet;
    bool m_RaggedOffsetDescSet;
};

} // namespace backend_api

} // namespace miopen
