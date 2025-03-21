from typing import Dict

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

from jaxfluids.stencils.reconstruction.central.central_2 import CentralSecondOrderReconstruction
from jaxfluids.stencils.reconstruction.central.central_adap_2 import CentralSecondOrderAdapReconstruction
from jaxfluids.stencils.reconstruction.central.central_4 import CentralFourthOrderReconstruction
from jaxfluids.stencils.reconstruction.central.central_adap_4 import CentralFourthOrderAdapReconstruction
from jaxfluids.stencils.reconstruction.central.central_6 import CentralSixthOrderReconstruction
from jaxfluids.stencils.reconstruction.central.central_adap_6 import CentralSixthOrderAdapReconstruction
from jaxfluids.stencils.reconstruction.central.central_8 import CentralEighthOrderReconstruction
from jaxfluids.stencils.reconstruction.central.central_adap_8 import CentralEighthOrderAdapReconstruction

CENTRAL_RECONSTRUCTION_DICT: Dict[str, SpatialReconstruction] = {
    "CENTRAL2":         CentralSecondOrderReconstruction,
    "CENTRAL2-ADAP":    CentralSecondOrderAdapReconstruction,
    "CENTRAL4":         CentralFourthOrderReconstruction,
    "CENTRAL4-ADAP":    CentralFourthOrderAdapReconstruction,
    "CENTRAL6":         CentralSixthOrderReconstruction,
    "CENTRAL6-ADAP":    CentralSixthOrderAdapReconstruction,
    "CENTRAL8":         CentralEighthOrderReconstruction,
    "CENTRAL8-ADAP":    CentralEighthOrderAdapReconstruction,
}