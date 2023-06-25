### Namespace: ark

#### Functions

- `void srand(int seed = -1)`: Sets the random seed.
- `int rand()`: Generates a random integer.

#### Initialization

- `void init()`: Initializes an ark program. Call this function to clean up the shared memory directory.

#### Data Types

- `typedef long long int DimType`: Data type for dimensions.

#### Constants

- `DIMS_LEN`: Maximum number of dimensions (4).
- `NO_DIM`: Constant representing an invalid dimension (-1).

#### Dims

- `struct Dims`: Represents an up-to-`DIMS_LEN`-dimensional vector.

  - **Constructors**:
    - `Dims(DimType d0 = NO_DIM, DimType d1 = NO_DIM, DimType d2 = NO_DIM, DimType d3 = NO_DIM)`: Constructs a `Dims` object with given four dimensions.
    - `Dims(const Dims &dims_)`: Constructs a `Dims` object by copying another `Dims` object.
    - `Dims(const std::vector<DimType> &vec)`: Constructs a `Dims` object from a vector. If the vector is shorter than `DIMS_LEN`, appends `NO_DIM`s. Raises an error if the vector is longer than `DIMS_LEN`.
  
  - **Methods**:
    - `DimType size() const`: Returns the volume of dimensions. If the dimensions are invalid, returns -1.
    - `int ndims() const`: Returns the number of valid dimensions.
    - `Dims dims4() const`: Returns a new `Dims` object with 4 valid dimensions by prepending 1s.
    - `bool is_no_dim() const`: Returns true if the dimensions are empty.
    - `bool is_invalid() const`: Returns true if the dimensions are invalid.
  
  - **Operators**:
    - `DimType &operator[](DimType idx)`: Returns a reference to the dimension at the specified index.
    - `const DimType &operator[](DimType idx) const`: Returns a constant reference to the dimension at the specified index.
    - `constexpr Dims &operator=(const Dims &) = default`: Default assignment operator.
  
  - **Friend Functions**:
    - `bool operator==(const Dims &a, const Dims &b)`: Equality operator.
    - `bool operator!=(const Dims &a, const Dims &b)`: Inequality operator.
    - `std::ostream &operator<<(std::ostream &os, const Dims &dims)`: Stream insertion operator.
  
#### Tensor

<!-- struct Tensor
{
    // Tensor constructor
    Tensor(const Dims &shape, TensorType type, TensorBuf *buf,
           const Dims &ldims, const Dims &offs, const Dims &pads, bool exported,
           bool imported, int id, const std::string &name);
    Tensor(const Tensor &) = default;

    void update_pads(const std::vector<DimType> &pads);
    // Offset to the element [i0][i1][i2][i3] of this tensor in the TensorBuf.
    DimType offset(DimType i0 = 0, DimType i1 = 0, DimType i2 = 0,
                   DimType i3 = 0) const;
    // Number of elements in the tensor excluding padding.
    DimType size() const;
    // Number of dimensions in the tensor.
    int ndims() const;
    // Shape of the tensor including padding.
    Dims padded_shape() const;
    // Number of bytes of each element in the tensor.
    unsigned int type_bytes() const;
    // Number of bytes of the tensor.
    DimType shape_bytes() const;
    // Should be the same as the number of bytes of the TensorBuf.
    DimType ldims_bytes() const;
    // Offset in bytes.
    DimType offset_bytes(DimType i0 = 0, DimType i1 = 0, DimType i2 = 0,
                         DimType i3 = 0) const;
    // TODO: deprecate this function.
    bool is_sequential() const;

    // TensorBuf that this tensor is associated with
    TensorBuf *buf;
    // Data type of each element in the tensor
    TensorType type;
    // Shape of the tensor
    Dims shape;
    // Leading dimensions of the underlying data array
    Dims ldims;
    // Offset of the tensor in the underlying data array
    Dims offs;
    // Unit dimensions of the underlying data array. ldims[x] should be always
    // divided by udims[x].
    Dims pads;
    // Whether this tensor is accessed by remote devices
    bool exported;
    // if imported is true, the tensor is imported from another GPU and don't
    // need to allocate a TensorBuf for it.
    bool imported;
    // Unique id of this tensor
    int id;
    // Name of this tensor
    const std::string name;
}; -->

