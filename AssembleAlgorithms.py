from LinearSystem import LinearSystem
from FieldsHolder import FieldArray, DimType,MAX_DIM, FieldNames
from MeshObject import MeshObject
from enum import Enum

class Boundary(Enum):
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3

class AssembleSystemBase:
    def __init__(self, name, fieldsHolder, linearSystem, meshObject):
        self.name_ = name
        self.fieldsHolder_ : FieldArray = fieldsHolder
        self.linearSystem_ : LinearSystem = linearSystem
        self.myMeshObject_ : MeshObject = meshObject
    def zero(self):
        self.linearSystem_.zero()
    def assemble(self):
        raise NotImplementedError("Must implement assemble() in subclass")
    def __repr__(self):
        return f"AssembleSystemBase(name='{self.name_}', field='{self.fieldsHolder_.get_name()}', linearSystem='{self.linearSystem_}')"

class AssembleCellVectorTimeTerm(AssembleSystemBase):
    def __init__(self, name, fieldsHolderNew, fieldsHolderOld, linearSystem, meshObject, dt):
        super().__init__(name, fieldsHolderNew, linearSystem, meshObject)
        self.dt_ = dt
        self.fieldHolderOld_ = fieldsHolderOld
        assert fieldsHolderNew.get_type() == DimType.VECTOR, "AssembleCellVectorTimeTerm requires a vector field for state NEW"
        assert fieldsHolderOld.get_type() == DimType.VECTOR, "AssembleCellVectorTimeTerm requires a vector field for state OLD"
    def assemble(self):
        for cell in self.myMeshObject_.get_cells():
            cellID = cell.get_flat_id()
            cellVolume = cell.get_volume()
            for comp in range(self.fieldsHolder_.get_num_components()):
                lhsFactor = cellVolume/self.dt_
                rowIndex = cellID*self.fieldsHolder_.get_num_components() + comp
                value = (self.fieldsHolder_.get_data()[rowIndex] - self.fieldHolderOld_.get_data()[rowIndex])
                self.linearSystem_.add_lhs(rowIndex, rowIndex, -lhsFactor)
                self.linearSystem_.add_rhs(rowIndex, value*lhsFactor)

class AssembleCellVectorPressureGradientToLinSystem(AssembleSystemBase):
    def __init__(self, name, fieldsHolderNew, linearSystem, meshObject):
        super().__init__(name, fieldsHolderNew, linearSystem, meshObject)
        assert fieldsHolderNew.get_type() == DimType.VECTOR, "AssembleCellVectorPressureGradient requires a vector field for state NEW"
        assert fieldsHolderNew.get_name() == FieldNames.GRAD_PRESSURE.value, "AssembleCellVectorPressureGradient requires a field named 'grad_pressure' for the pressure gradient values"
    def assemble(self):
        for cell in self.myMeshObject_.get_cells():
            cellID = cell.get_flat_id()
            cellVolume = cell.get_volume()
            for comp in range(self.fieldsHolder_.get_num_components()):
                rowIndex = cellID*self.fieldsHolder_.get_num_components() + comp
                value = (self.fieldsHolder_.get_data()[rowIndex])*cellVolume
                self.linearSystem_.add_rhs(rowIndex, value)

class AssembleInteriorVectorAdvectionToLinSystem(AssembleSystemBase):
    def __init__(self, name, fieldsHolder, linearSystem, meshObject, diffusionCoeff=1.0):
        super().__init__(name, fieldsHolder, linearSystem, meshObject)
        self.diffusionCoeff_ = diffusionCoeff
        assert fieldsHolder.get_type() == DimType.VECTOR, "AssembleInteriorVectorAdvectionToLinSystem requires a vector field"
    def assemble(self):
        # Loop over internal faces and assemble contributions to the linear system
        for face in self.myMeshObject_.get_internal_faces():
            leftCellID = face.get_left_cell()
            rightCellID = face.get_right_cell()
            faceFlux = face.massFlux_
            mdot_L = (faceFlux + abs(faceFlux))/2.0
            mdot_R = (faceFlux - abs(faceFlux))/2.0
            fieldLeft = [self.fieldsHolder_.get_data()[leftCellID*MAX_DIM + i] for i in range(MAX_DIM)]
            fieldRight = [self.fieldsHolder_.get_data()[rightCellID*MAX_DIM + i] for i in range(MAX_DIM)]
            rhs = [fieldLeft[i]*mdot_L + fieldRight[i]*mdot_R for i in range(MAX_DIM)]

            # Add contribution to the RHS/LHS
            for i in range(MAX_DIM):
                rowIndexLeft = leftCellID*MAX_DIM+i
                rowIndexRight = rightCellID*MAX_DIM+i
                self.linearSystem_.add_rhs(rowIndexLeft, rhs[i])
                self.linearSystem_.add_rhs(rowIndexRight, -rhs[i])
                self.linearSystem_.add_lhs(rowIndexLeft, rowIndexLeft, -mdot_L)  
                self.linearSystem_.add_lhs(rowIndexLeft, rowIndexRight, -mdot_R)  
                self.linearSystem_.add_lhs(rowIndexRight, rowIndexLeft, mdot_L)  
                self.linearSystem_.add_lhs(rowIndexRight, rowIndexLeft, mdot_R)  

    def __repr__(self):
        return super().__repr__()


class AssembleInteriorVectorDiffusionToLinSystem(AssembleSystemBase):
    def __init__(self, name, fieldsHolder, linearSystem, meshObject, diffusionCoeff=1.0):
        super().__init__(name, fieldsHolder, linearSystem, meshObject)
        self.diffusionCoeff_ = diffusionCoeff
        assert fieldsHolder.get_type() == DimType.VECTOR, "AssembleInteriorVectorDiffusionToLinSystem requires a vector field"
    def assemble(self):
        # Loop over internal faces and assemble contributions to the linear system
        for face in self.myMeshObject_.get_internal_faces():
            leftCellID = face.get_left_cell()
            rightCellID = face.get_right_cell()
            faceArea = face.get_area()
            normal = face.get_normal_vector()
            cellLeft = self.myMeshObject_.get_cell_by_flat_id(leftCellID)
            cellRight = self.myMeshObject_.get_cell_by_flat_id(rightCellID)
            distance = [cellRight.get_centroid()[i] - cellLeft.get_centroid()[i] for i in range(MAX_DIM)]  # Vector from left cell to right cell
            normalLength = normal[0]*distance[0] + normal[1]*distance[1]  # Dot product of normal and distance vector
            lhsFactor = faceArea/normalLength*self.diffusionCoeff_
            gradDotArea = [(self.fieldsHolder_.get_data()[rightCellID*MAX_DIM + i] - self.fieldsHolder_.get_data()[leftCellID*MAX_DIM + i])*lhsFactor for i in range(MAX_DIM)]

            # Add contribution to the RHS/LHS
            for i in range(MAX_DIM):
                self.linearSystem_.add_rhs(leftCellID*MAX_DIM + i, -gradDotArea[i])
                self.linearSystem_.add_rhs(rightCellID*MAX_DIM + i, gradDotArea[i])
                self.linearSystem_.add_lhs(leftCellID*MAX_DIM + i, rightCellID*MAX_DIM + i, lhsFactor)  
                self.linearSystem_.add_lhs(leftCellID*MAX_DIM + i, leftCellID*MAX_DIM + i, -lhsFactor)  
                self.linearSystem_.add_lhs(rightCellID*MAX_DIM + i, rightCellID*MAX_DIM + i, -lhsFactor)  
                self.linearSystem_.add_lhs(rightCellID*MAX_DIM + i, leftCellID*MAX_DIM + i, lhsFactor)  

    def __repr__(self):
        return super().__repr__()
    
class AssembleDirichletBoundaryVectorDiffusionToLinSystem(AssembleSystemBase):
    def __init__(self, name, fieldsHolder, linearSystem, meshObject, boundaryType, boundaryValue, diffusionCoeff=1.0):
        super().__init__(name, fieldsHolder, linearSystem, meshObject)
        assert type(boundaryType) == Boundary, "boundaryType must be an instance of the Boundary enum"
        assert fieldsHolder.get_type() == DimType.VECTOR, "AssembleDirichletBoundaryVectorDiffusionToLinSystem requires a vector field"
        assert type(boundaryValue) == list and len(boundaryValue) == MAX_DIM, "boundaryValue must be a list of length equal to the number of dimensions"
        self.myFaceIterable_ = None
        if boundaryType == Boundary.LEFT:
            self.myFaceIterable_ = meshObject.get_left_boundary()
        elif boundaryType == Boundary.RIGHT:
            self.myFaceIterable_ = meshObject.get_right_boundary()
        elif boundaryType == Boundary.TOP:
            self.myFaceIterable_ = meshObject.get_top_boundary()
        elif boundaryType == Boundary.BOTTOM:
            self.myFaceIterable_ = meshObject.get_bottom_boundary()  
        assert self.myFaceIterable_ is not None, "Invalid boundary type specified"

        self.diffusionCoeff_ = diffusionCoeff
        self.boundaryValue_ = boundaryValue  # The Dirichlet value (e.g., Temperature at wall)

    def assemble(self):
        """Loop over boundary faces and assemble diffusive flux."""
        for face in self.myFaceIterable_:
            cellID = face.get_left_cell()
            faceArea = face.get_area()
            normal = face.get_normal_vector()
            
            owningCell = self.myMeshObject_.get_cell_by_flat_id(cellID)
            
            # Vector from cell centroid to face centroid
            distanceVec = [face.get_face_center()[i] - owningCell.get_centroid()[i] for i in range(2)]
            
            # This is 'd_f' in the flux equation: J = -D * (phi_boundary - phi_cell) / d_f
            normalDistance = (normal[0]*distanceVec[0] + normal[1]*distanceVec[1])
            lhsFactor = (self.diffusionCoeff_ * faceArea) / normalDistance
            gradDotArea = [(self.boundaryValue_[i] - self.fieldsHolder_.get_data()[cellID*MAX_DIM + i])*lhsFactor for i in range(MAX_DIM)]

            for i in range(MAX_DIM):
                self.linearSystem_.add_rhs(cellID*MAX_DIM+i, -gradDotArea[i])
                self.linearSystem_.add_lhs(cellID*MAX_DIM+i, cellID*MAX_DIM+i, -lhsFactor)

class AssembleInteriorPressurePoissonSystem(AssembleSystemBase):
    def __init__(self, name, fieldsHolder, linearSystem, meshObject, dt):
        super().__init__(name, fieldsHolder, linearSystem, meshObject)
        self.dt_ = dt
        assert fieldsHolder.get_type() == DimType.SCALAR, "AssembleInteriorPressurePoissonSystem requires a scalar field"
        assert fieldsHolder.get_name() == FieldNames.PRESSURE.value, "AssembleInteriorPressurePoissonSystem requires a field named 'Pressure' for the pressure values"
    def assemble(self):
        # Loop over internal faces and assemble contributions to the linear system
        for face in self.myMeshObject_.get_internal_faces():
            leftCellID = face.get_left_cell()
            rightCellID = face.get_right_cell()
            faceArea = face.get_area()
            normal = face.get_normal_vector()
            cellLeft = self.myMeshObject_.get_cell_by_flat_id(leftCellID)
            cellRight = self.myMeshObject_.get_cell_by_flat_id(rightCellID)
            distance = [cellRight.get_centroid()[i] - cellLeft.get_centroid()[i] for i in range(MAX_DIM)]  # Vector from left cell to right cell
            normalLength = normal[0]*distance[0] + normal[1]*distance[1]  # Dot product of normal and distance vector
            lhsFactor = self.dt_*faceArea/normalLength
            massFlux = face.massFlux_

            # Add contribution to the RHS
            self.linearSystem_.add_rhs(leftCellID, massFlux) 
            self.linearSystem_.add_rhs(rightCellID, -massFlux)
            # Add contribution to the LHS
            self.linearSystem_.add_lhs(leftCellID, rightCellID, lhsFactor)  
            self.linearSystem_.add_lhs(leftCellID, leftCellID, -lhsFactor)  
            self.linearSystem_.add_lhs(rightCellID, rightCellID, -lhsFactor)  
            self.linearSystem_.add_lhs(rightCellID, leftCellID, lhsFactor)  

class AssembleInteriorScalarDiffusionToLinSystem(AssembleSystemBase):
    def __init__(self, name, fieldsHolder, linearSystem, meshObject, diffusionCoeff=1.0):
        super().__init__(name, fieldsHolder, linearSystem, meshObject)
        self.diffusionCoeff_ = diffusionCoeff
        assert fieldsHolder.get_type() == DimType.SCALAR, "AssembleInteriorScalarDiffusionToLinSystem requires a scalar field"
    def assemble(self):
        # Loop over internal faces and assemble contributions to the linear system
        for face in self.myMeshObject_.get_internal_faces():
            leftCellID = face.get_left_cell()
            rightCellID = face.get_right_cell()
            faceArea = face.get_area()
            normal = face.get_normal_vector()
            cellLeft = self.myMeshObject_.get_cell_by_flat_id(leftCellID)
            cellRight = self.myMeshObject_.get_cell_by_flat_id(rightCellID)
            distance = [cellRight.get_centroid()[i] - cellLeft.get_centroid()[i] for i in range(MAX_DIM)]  # Vector from left cell to right cell
            normalLength = normal[0]*distance[0] + normal[1]*distance[1]  # Dot product of normal and distance vector
            lhsFactor = faceArea/normalLength*self.diffusionCoeff_
            gradDotArea = (self.fieldsHolder_.get_data()[rightCellID] - self.fieldsHolder_.get_data()[leftCellID])*lhsFactor

            # Add contribution to the RHS
            self.linearSystem_.add_rhs(leftCellID, -gradDotArea) 
            self.linearSystem_.add_rhs(rightCellID, gradDotArea)
            # Add contribution to the LHS
            self.linearSystem_.add_lhs(leftCellID, rightCellID, lhsFactor)  
            self.linearSystem_.add_lhs(leftCellID, leftCellID, -lhsFactor)  
            self.linearSystem_.add_lhs(rightCellID, rightCellID, -lhsFactor)  
            self.linearSystem_.add_lhs(rightCellID, leftCellID, lhsFactor)  

    def __repr__(self):
        return super().__repr__()
    
class AssembleDirichletBoundaryScalarDiffusionToLinSystem(AssembleSystemBase):
    def __init__(self, name, fieldsHolder, linearSystem, meshObject, boundaryType, boundaryValue, diffusionCoeff=1.0):
        super().__init__(name, fieldsHolder, linearSystem, meshObject)
        assert type(boundaryType) == Boundary, "boundaryType must be an instance of the Boundary enum"
        assert fieldsHolder.get_type() == DimType.SCALAR, "AssembleDirichletBoundaryScalarDiffusionToLinSystem requires a scalar field"
        self.myFaceIterable_ = None
        if boundaryType == Boundary.LEFT:
            self.myFaceIterable_ = meshObject.get_left_boundary()
        elif boundaryType == Boundary.RIGHT:
            self.myFaceIterable_ = meshObject.get_right_boundary()
        elif boundaryType == Boundary.TOP:
            self.myFaceIterable_ = meshObject.get_top_boundary()
        elif boundaryType == Boundary.BOTTOM:
            self.myFaceIterable_ = meshObject.get_bottom_boundary()  
        assert self.myFaceIterable_ is not None, "Invalid boundary type specified"

        self.diffusionCoeff_ = diffusionCoeff
        self.boundaryValue_ = boundaryValue  # The Dirichlet value (e.g., Temperature at wall)

    def assemble(self):
        """Loop over boundary faces and assemble diffusive flux."""
        for face in self.myFaceIterable_:
            cellID = face.get_left_cell()
            faceArea = face.get_area()
            normal = face.get_normal_vector()
            
            owningCell = self.myMeshObject_.get_cell_by_flat_id(cellID)
            
            # Vector from cell centroid to face centroid
            distanceVec = [face.get_face_center()[i] - owningCell.get_centroid()[i] for i in range(2)]
            
            # This is 'd_f' in the flux equation: J = -D * (phi_boundary - phi_cell) / d_f
            normalDistance = (normal[0]*distanceVec[0] + normal[1]*distanceVec[1])
            lhsFactor = (self.diffusionCoeff_ * faceArea) / normalDistance
            gradDotArea = (self.boundaryValue_ - self.fieldsHolder_.get_data()[cellID])*lhsFactor
            self.linearSystem_.add_rhs(cellID, -gradDotArea)
            self.linearSystem_.add_lhs(cellID, cellID, -lhsFactor)