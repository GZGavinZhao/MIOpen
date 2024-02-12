#include <miopen/backend_api.hpp>
#include <miopen/miopen.h>
#include <miopen/errors.hpp>
#include <new>

extern "C" miopenStatus_t
miopenBackendCreateDescriptor(miopenBackendDescriptorType_t descriptorType,
                              miopenBackendDescriptor_t* descriptor)
{
    switch(descriptorType)
    {
    case MIOPEN_BACKEND_TENSOR_DESCRIPTOR:
        // TODO: add logging
        // TODO: get rid of unnecessary throws
        return miopen::try_(
            [&] {
                miopen::backend_api::BackendTensorDescriptor* newDescriptor =
                    new(std::nothrow) miopen::backend_api::BackendTensorDescriptor();

                if(!newDescriptor)
                {
                    throw miopen::Exception(miopenStatusAllocFailed);
                }

                try
                {
                    if(newDescriptor->Status() != miopenStatusSuccess)
                    {
                        miopenStatus_t status = newDescriptor->Status();
                        throw miopen::Exception(status);
                    }

                    miopen::deref(descriptor) = newDescriptor;
                }
                catch(...)
                {
                    delete newDescriptor;
                    throw;
                }
            },
            false);

    default: return miopenStatus_t::miopenStatusUnsupportedOp;
    }
}

extern "C" miopenStatus_t miopenBackendSetAttribute(miopenBackendDescriptor_t descriptor,
                                                    miopenBackendAttributeName_t attributeName,
                                                    miopenBackendAttributeType_t attributeType,
                                                    int64_t elementCount,
                                                    void* arrayOfElements)
{
    return miopen::try_(
        [&] {
            miopen::backend_api::BackendDescriptor* theDescriptor =
                static_cast<miopen::backend_api::BackendDescriptor*>(descriptor);

            miopenStatus_t status = theDescriptor->SetAttribute(
                attributeName, attributeType, elementCount, arrayOfElements);

            if(status != miopenStatusSuccess)
            {
                throw miopen::Exception(status);
            }
        },
        false);
}

extern "C" miopenStatus_t miopenBackendFinalize(miopenBackendDescriptor_t descriptor)
{
    return miopen::try_(
        [&] {
            miopen::backend_api::BackendDescriptor* theDescriptor =
                static_cast<miopen::backend_api::BackendDescriptor*>(descriptor);

            miopenStatus_t status = theDescriptor->Finalize();

            if(status != miopenStatusSuccess)
            {
                throw miopen::Exception(status);
            }
        },
        false);
}

namespace miopen {

namespace backend_api {

BackendTensorDescriptor::BackendTensorDescriptor()
    : m_Descriptor(nullptr),
      m_VectorCount(1),
      m_IsVirtual(false),
      m_UniqueIdSet(false),
      m_DataTypeSet(false),
      m_ByteAlignmentSet(false),
      m_DimensionsSet(false),
      m_StridesSet(false),
      m_VectorCountSet(false),
      m_VectorizedDimensionSet(false),
      m_RaggedOffsetDescSet(false)
{
    m_Status = miopenCreateTensorDescriptor(&m_Descriptor);
}

BackendTensorDescriptor::~BackendTensorDescriptor()
{
    if(m_Descriptor)
    {
        miopenDestroyTensorDescriptor(m_Descriptor);
    }
}

miopenStatus_t BackendTensorDescriptor::SetAttribute(miopenBackendAttributeName_t attributeName,
                                                     miopenBackendAttributeType_t attributeType,
                                                     int64_t elementCount,
                                                     void* arrayOfElements)
{
    if(m_IsFinalized)
    {
        return miopenStatusNotInitialized;
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_TENSOR_UNIQUE_ID:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            m_UniqueId    = *reinterpret_cast<int64_t*>(arrayOfElements);
            m_UniqueIdSet = true;
            return miopenStatusSuccess;
        }
        else
        {
            return miopenStatusBadParm;
        }

    case MIOPEN_ATTR_TENSOR_DATA_TYPE:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && elementCount == 1)
        {
            m_DataType    = *reinterpret_cast<miopenDataType_t*>(arrayOfElements);
            m_DataTypeSet = true;
            return miopenStatusSuccess;
        }
        else
        {
            return miopenStatusBadParm;
        }

    case MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            m_ByteAlignment    = *reinterpret_cast<int64_t*>(arrayOfElements);
            m_ByteAlignmentSet = true;
            return miopenStatusSuccess;
        }
        else
        {
            return miopenStatusBadParm;
        }

    case MIOPEN_ATTR_TENSOR_DIMENSIONS:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 4)
        {
            m_Dimensions =
                std::vector<int64_t>(reinterpret_cast<int64_t*>(arrayOfElements),
                                     reinterpret_cast<int64_t*>(arrayOfElements) + elementCount);
            m_DimensionsSet = true;
            return miopenStatusSuccess;
        }
        else
        {
            return miopenStatusBadParm;
        }

    case MIOPEN_ATTR_TENSOR_STRIDES:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 4)
        {
            m_Strides =
                std::vector<int64_t>(reinterpret_cast<int64_t*>(arrayOfElements),
                                     reinterpret_cast<int64_t*>(arrayOfElements) + elementCount);
            m_StridesSet = true;
            return miopenStatusSuccess;
        }
        else
        {
            return miopenStatusBadParm;
        }

    case MIOPEN_ATTR_TENSOR_VECTOR_COUNT:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            m_VectorCount    = *reinterpret_cast<int64_t*>(arrayOfElements);
            m_VectorCountSet = true;
            return miopenStatusSuccess;
        }
        else
        {
            return miopenStatusBadParm;
        }

    case MIOPEN_ATTR_TENSOR_VECTORIZED_DIMENSION:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            m_VectorizedDimension    = *reinterpret_cast<int64_t*>(arrayOfElements);
            m_VectorizedDimensionSet = true;
            return miopenStatusSuccess;
        }
        else
        {
            return miopenStatusBadParm;
        }

    case MIOPEN_ATTR_TENSOR_IS_VIRTUAL:
        if(attributeType == MIOPEN_TYPE_BOOLEAN && elementCount == 1)
        {
            m_IsVirtual = *reinterpret_cast<bool*>(arrayOfElements);
            return miopenStatusSuccess;
        }
        else
        {
            return miopenStatusBadParm;
        }

    case MIOPEN_ATTR_TENSOR_RAGGED_OFFSET_DESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            m_RaggedOffsetDesc    = *reinterpret_cast<miopenBackendDescriptor_t*>(arrayOfElements);
            m_RaggedOffsetDescSet = true;
            return miopenStatusSuccess;
        }
        else
        {
            return miopenStatusBadParm;
        }

    default: return miopenStatusBadParm;
    }
}

miopenStatus_t BackendTensorDescriptor::Finalize()
{
    if(!m_UniqueIdSet || !m_DataTypeSet || !m_ByteAlignmentSet || !m_DimensionsSet ||
       !m_StridesSet || (m_VectorCountSet && !m_VectorizedDimensionSet))
    {
        return miopenStatusBadParm;
    }

    if(m_Dimensions.size() != 4 || m_Strides.size() != 4)
    {
        return miopenStatusBadParm;
    }

    return miopenSet4dTensorDescriptorEx(m_Descriptor,
                                         m_DataType,
                                         m_Dimensions[0],
                                         m_Dimensions[1],
                                         m_Dimensions[2],
                                         m_Dimensions[3],
                                         m_Strides[0],
                                         m_Strides[1],
                                         m_Strides[2],
                                         m_Strides[3]);
}

} // namespace backend_api

} // namespace miopen
