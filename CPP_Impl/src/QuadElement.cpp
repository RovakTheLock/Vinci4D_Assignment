#include "QuadElement.h"

namespace Vinci4D {

Face::Face(int faceId, int leftCellIdx, int rightCellIdx,
           std::array<double, 2> faceCoords, std::array<double, 2> normalVector)
    : id_(faceId),
      leftCellFlatId_(leftCellIdx),
      rightCellFlatId_(rightCellIdx),
      faceCoords_(faceCoords),
      normalVector_(normalVector),
      area_(0.0),
      isBoundary_(rightCellIdx < 0),
      massFlux_(0.0) {}

Cell::Cell(int flatId, double volume, std::array<int, 2> indices, std::array<double, 2> centroid)
    : flatId_(flatId),
      localId_(-1),
      volume_(volume),
      indices_(indices),
      centroid_(centroid) {}

} // namespace Vinci4D
