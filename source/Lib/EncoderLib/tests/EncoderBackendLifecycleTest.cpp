#include "EncLibCommon.h"
#include "CudaBackend/ComputeBackend.h"

#include <cstdlib>
#include <functional>
#include <iostream>

namespace
{

bool throws(const std::function<void()> &operation)
{
  try
  {
    operation();
  }
  catch (const std::exception &)
  {
    return true;
  }
  return false;
}

int fail(const char *message)
{
  std::cerr << message << '\n';
  return EXIT_FAILURE;
}

}   // namespace

int main()
{
  EncLibCommon cpuBackend;
  cpuBackend.configureComputeBackend({ vtm::ComputeBackend::CPU, 0 });
  cpuBackend.acquireComputeBackend();
  cpuBackend.releaseComputeBackend();
  cpuBackend.acquireComputeBackend();
  cpuBackend.releaseComputeBackend();

  EncLibCommon failingBackend;
  failingBackend.configureComputeBackend({ vtm::ComputeBackend::CUDA, 999 });
  if (!throws([&failingBackend]() { failingBackend.acquireComputeBackend(); }))
  {
    return fail("Invalid CUDA backend acquisition unexpectedly succeeded");
  }
  if (!throws([&failingBackend]() { failingBackend.releaseComputeBackend(); }))
  {
    return fail("Failed acquisition leaked an encoder backend owner");
  }
  if (!throws([&failingBackend]() { failingBackend.acquireComputeBackend(); }))
  {
    return fail("Invalid CUDA backend acquisition unexpectedly succeeded on retry");
  }
  if (!throws([&failingBackend]() { failingBackend.releaseComputeBackend(); }))
  {
    return fail("Retried failed acquisition leaked an encoder backend owner");
  }

  return EXIT_SUCCESS;
}
