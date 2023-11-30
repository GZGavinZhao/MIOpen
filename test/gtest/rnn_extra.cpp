/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "../rnn_vanilla.hpp"
#include "get_handle.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

static bool SkipTest(void) { return miopen::IsDisabled(ENV(MIOPEN_TEST_ALL)); }

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class ConfigWithFloat : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = ConfigWithFloat::GetParam(); break;
    case miopenHalf:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenDouble:
        FAIL() << "miopenHalf, miopenInt8, miopenBFloat16, miopenInt8x4, miopenInt32, miopenDouble "
                  "data type not supported by "
                  "rnn_vanilla test";

    default: params = ConfigWithFloat::GetParam();
    }

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });

        testing::internal::CaptureStderr();
        test_drive<rnn_vanilla_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    if(devName == "gfx900" || devName == "gfx906" || devName == "gfx908" || devName == "gfx90a" ||
       miopen::StartsWith(devName, "gfx94") || miopen::StartsWith(devName, "gfx103") ||
       miopen::StartsWith(devName, "gfx110"))
        return true;
    else
        return false;
}

TEST_P(ConfigWithFloat, FloatTest)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest())
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

std::vector<std::string> GetTestCases(void)
{
    std::string commonFlags =
        " --verbose --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 "
        "--hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0";
    std::string dir0   = " -dir-mode 0";
    std::string dir1   = " -dir-mode 1";
    std::string rnn0   = " --rnn-mode 0";
    std::string rnn1   = " --rnn-mode 1";
    std::string no_hx  = " --no-hx";
    std::string no_hy  = " --no-hy";
    std::string no_dhx = " --no-dhx";
    std::string no_dhy = " --no-dhy";

    const std::vector<std::string> test_cases = {
        // clang-format off
    	{commonFlags + dir0 + rnn0 + no_hx},
		{commonFlags + dir0 + rnn0 + no_dhy},
		{commonFlags + dir0 + rnn0 + no_hx + no_dhy},
		{commonFlags + dir0 + rnn1 + no_hx},
		{commonFlags + dir0 + rnn1 + no_dhy},
		{commonFlags + dir0 + rnn1 + no_hx + no_dhy},
		{commonFlags + dir1 + rnn0 + no_hx},
		{commonFlags + dir1 + rnn0 + no_dhy},
		{commonFlags + dir1 + rnn0 + no_hx + no_dhy},
		{commonFlags + dir1 + rnn1 + no_hx},
		{commonFlags + dir1 + rnn1 + no_dhy},
		{commonFlags + dir1 + rnn1 + no_hx + no_dhy},
		{commonFlags + dir0 + rnn0 + no_hy},
		{commonFlags + dir0 + rnn0 + no_dhx},
		{commonFlags + dir0 + rnn0 + no_hy + no_dhx},
		{commonFlags + dir0 + rnn1 + no_hy},
		{commonFlags + dir0 + rnn1 + no_dhx},
		{commonFlags + dir0 + rnn1 + no_hy + no_dhx},
		{commonFlags + dir1 + rnn0 + no_hy},
		{commonFlags + dir1 + rnn0 + no_dhx},
		{commonFlags + dir1 + rnn0 + no_hy + no_dhx},
		{commonFlags + dir1 + rnn1 + no_hy},
		{commonFlags + dir1 + rnn1 + no_dhx},
		{commonFlags + dir1 + rnn1 + no_hy + no_dhx},
		{commonFlags + dir0 + rnn0 + no_hx + no_dhy + no_hy + no_dhx},
		{commonFlags + dir0 + rnn1 + no_hx + no_dhy + no_hy + no_dhx},
		{commonFlags + dir1 + rnn0 + no_hx + no_dhy + no_hy + no_dhx},
		{commonFlags + dir1 + rnn1 + no_hx + no_dhy + no_hy + no_dhx}
        // clang-format on
    };

    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(ConvTrans, ConfigWithFloat, testing::Values(GetTestCases()));
