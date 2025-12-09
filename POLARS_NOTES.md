# Polars Usage Notes & Gotchas

Key learnings and differences from vanilla Python when working with Polars.

## List Operations

### `list.slice()` Parameters

**GOTCHA**: `list.slice(offset, length)` takes a **length** as the second parameter, NOT a stop index like Python's slice.

```python
# Python
my_list[1:3]  # Start at 1, stop at 3 (exclusive)

# Polars - WRONG interpretation
pl.col("list").list.slice(1, 3)  # Start at 1, take 3 elements (not stop at 3!)

# Polars - Correct
pl.col("list").list.slice(1, 2)  # Start at 1, take 2 elements -> equivalent to [1:3]
```

**Documentation**: <https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.list.slice.html>

### `list.head()` with Negative Values

**GOTCHA**: `list.head(-1)` does NOT work like Python's `[:-1]` (all but last element).

```python
# Python
my_list[:-1]  # All but the last element

# Polars - INCORRECT
pl.col("list").list.head(-1)  # Does NOT behave like Python's [:-1]

# Polars - CORRECT approach
pl.col("list").list.slice(0, pl.col("list").list.len() - 1)
```

**Better**: Calculate the length explicitly and slice:

```python
df = df.with_columns(pl.col("list").list.len().alias("len"))
df = df.with_columns(
    pl.col("list").list.slice(0, pl.col("len") - 1).alias("all_but_last")
)
```

### `list.arg_max()` and Ties

**GOTCHA**: When there are multiple maximum values, `list.arg_max()` returns the **last** occurrence, not the first (unlike Python's `list.index()`).

```python
# Python
[9, 9, 5, 3].index(max([9, 9, 5, 3]))  # Returns 0 (first occurrence)

# Polars
pl.Series([[9, 9, 5, 3]]).list.arg_max()  # Returns 1 (last occurrence)
```

**Solution**: Use `str.find()` on the original string to find the first occurrence:

```python
# Find the first index of the max value by searching the original string
df = df.with_columns(
    pl.col("string_col").str.find(pl.col("max_value").cast(pl.String)).alias("first_idx")
)
```

## String to List Conversion

### `str.split("")` Creates Empty Strings

**GOTCHA**: Using `str.split("")` to split a string into characters creates empty strings at both the beginning and end.

```python
# Polars
pl.Series(["abc"]).str.split("")  # Returns [["", "a", "b", "c", ""]]

# Need to slice to remove empties
pl.Series(["abc"]).str.split("").list.slice(1, -1)  # But -1 is length, not stop!
```

**Better Approach**: Use `str.extract_all()` with regex:

```python
# Extract all digit characters cleanly
pl.col("string").str.extract_all(r"\d")  # Returns ["1", "2", "3"] for "123"

# Extract all characters (any character)
pl.col("string").str.extract_all(r".")  # Returns ["a", "b", "c"] for "abc"
```

## Casting and Data Types

### Int8 Range

`pl.Int8` can hold values from -128 to 127, which is sufficient for single digits (0-9) but verify your data range:

```python
# Safe for digits
.list.eval(pl.element().cast(pl.Int8))  # Fine for "0"-"9"

# Use Int64 if unsure
.list.eval(pl.element().cast(pl.Int64))
```

## Useful Patterns

### Converting String Digits to Integer List

```python
# Clean approach using str.extract_all
df = pl.DataFrame({"numbers": ["12345", "67890"]})
df = df.with_columns(
    pl.col("numbers")
    .str.extract_all(r"\d")
    .list.eval(pl.element().cast(pl.Int8))
    .alias("digits")
)
```

### Finding First Occurrence in Lists

When you need Python's `list.index()` behavior (first occurrence):

```python
# Instead of list.arg_max() which returns last occurrence
# Use string search on the original data
df = df.with_columns(
    pl.col("original_string")
    .str.find(pl.col("target_value").cast(pl.String))
    .alias("first_index")
)
```

### Getting Tail After Specific Index

```python
# Get all elements after index N
pl.col("list").list.slice(N + 1)  # Takes from N+1 to end (None is default)

# Get all elements after dynamic index
pl.col("list").list.slice(pl.col("dynamic_idx") + 1)
```

## Resources

- **User Guide**: <https://docs.pola.rs/user-guide/expressions/lists-and-arrays/>
- **API Reference - List**: <https://docs.pola.rs/api/python/stable/reference/expressions/list.html>
- **API Reference - String**: <https://docs.pola.rs/api/python/stable/reference/expressions/string.html>

## Key Takeaway

Polars list operations are optimized for columnar operations and may not behave exactly like Python's built-in list operations. When in doubt:

1. Check the API documentation for the exact parameter meanings
2. Use explicit length calculations for complex slicing
3. Test with small examples to verify behavior
4. Consider using string operations when dealing with character-level manipulation
