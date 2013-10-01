#include <panoptes/metadata.h>

using namespace panoptes;

bool metadata_ptrs::operator==(const metadata_ptrs & rhs) const {
    return adata == rhs.adata && vdata == rhs.vdata;
}

bool metadata_ptrs::operator!=(const metadata_ptrs & rhs) const {
    return adata != rhs.adata || vdata != rhs.vdata;
}
