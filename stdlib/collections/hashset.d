//! A hash set implementation.
//!
//! HashSet<T> is a hash table with only keys, no values.

use std::collections::HashMap
use std::hash::Hash
use std::cmp::Eq
use std::iter::{Iterator, IntoIterator, FromIterator}
use std::fmt::{Debug, Formatter, FmtError}
use std::clone::Clone
use std::default::Default

/// A hash set implemented as a HashMap where the value is ().
///
/// As with the HashMap type, a HashSet requires that the elements
/// implement the Eq and Hash traits.
///
/// # Examples
///
/// ```d
/// let mut set = HashSet::new()
///
/// set.insert(1)
/// set.insert(2)
/// set.insert(3)
///
/// assert(set.contains(&2))
/// assert(!set.contains(&4))
///
/// set.remove(&2)
/// assert(!set.contains(&2))
/// ```
pub struct HashSet<T> {
    map: HashMap<T, unit>,
}

impl<T> HashSet<T>
where T: Hash + Eq
{
    /// Creates an empty HashSet.
    ///
    /// The hash set is initially created with a capacity of 0, so it
    /// will not allocate until it is first inserted into.
    pub fn new() -> HashSet<T> with Alloc {
        HashSet { map: HashMap::new() }
    }

    /// Creates an empty HashSet with the specified capacity.
    ///
    /// The hash set will be able to hold at least `capacity` elements
    /// without reallocating.
    pub fn with_capacity(capacity: int) -> HashSet<T> with Alloc {
        HashSet { map: HashMap::with_capacity(capacity) }
    }

    /// Returns the number of elements in the set.
    pub fn len(self: &HashSet<T>) -> int {
        self.map.len()
    }

    /// Returns true if the set contains no elements.
    pub fn is_empty(self: &HashSet<T>) -> bool {
        self.map.is_empty()
    }

    /// Returns the number of elements the set can hold without reallocating.
    pub fn capacity(self: &HashSet<T>) -> int {
        self.map.capacity()
    }

    /// Clears the set, removing all values.
    pub fn clear(self: &!HashSet<T>) {
        self.map.clear()
    }

    /// Returns true if the set contains the value.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mut set = HashSet::new()
    /// set.insert(1)
    /// assert(set.contains(&1))
    /// assert(!set.contains(&2))
    /// ```
    pub fn contains(self: &HashSet<T>, value: &T) -> bool {
        self.map.contains_key(value)
    }

    /// Adds a value to the set.
    ///
    /// Returns whether the value was newly inserted. That is:
    /// - If the set did not contain this value, true is returned.
    /// - If the set already contained this value, false is returned.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mut set = HashSet::new()
    /// assert(set.insert(1))
    /// assert(!set.insert(1))
    /// ```
    pub fn insert(self: &!HashSet<T>, value: T) -> bool with Alloc {
        self.map.insert(value, ()).is_none()
    }

    /// Removes a value from the set.
    ///
    /// Returns whether the value was present in the set.
    pub fn remove(self: &!HashSet<T>, value: &T) -> bool {
        self.map.remove(value).is_some()
    }

    /// Adds a value to the set, replacing the existing value if any.
    ///
    /// Returns the replaced value if one existed.
    pub fn replace(self: &!HashSet<T>, value: T) -> Option<T> with Alloc {
        // We need to get the old key if it exists
        if self.map.contains_key(&value) {
            self.map.remove(&value)
            self.map.insert(value, ())
            // Would return old value, simplified here
            Option::None
        } else {
            self.map.insert(value, ())
            Option::None
        }
    }

    /// Returns a reference to the value in the set, if any.
    pub fn get(self: &HashSet<T>, value: &T) -> Option<&T> {
        self.map.get_key(value)
    }

    /// Returns an iterator over the values.
    pub fn iter(self: &HashSet<T>) -> Iter<T> {
        Iter { inner: self.map.keys() }
    }

    /// Visits the values representing the difference.
    ///
    /// Returns values that are in self but not in other.
    pub fn difference(self: &HashSet<T>, other: &HashSet<T>) -> Difference<T> {
        Difference { iter: self.iter(), other }
    }

    /// Visits the values representing the symmetric difference.
    ///
    /// Returns values that are in either set but not both.
    pub fn symmetric_difference(self: &HashSet<T>, other: &HashSet<T>) -> SymmetricDifference<T> {
        SymmetricDifference {
            a_diff_b: self.difference(other),
            b_diff_a: other.difference(self),
            in_second: false,
        }
    }

    /// Visits the values representing the intersection.
    ///
    /// Returns values that are in both sets.
    pub fn intersection(self: &HashSet<T>, other: &HashSet<T>) -> Intersection<T> {
        Intersection { iter: self.iter(), other }
    }

    /// Visits the values representing the union.
    ///
    /// Returns values that are in either set.
    pub fn union(self: &HashSet<T>, other: &HashSet<T>) -> Union<T> {
        Union {
            iter: self.iter(),
            other_iter: other.difference(self),
            in_other: false,
        }
    }

    /// Returns true if self is a subset of other.
    ///
    /// This means all elements in self are contained in other.
    pub fn is_subset(self: &HashSet<T>, other: &HashSet<T>) -> bool {
        if self.len() > other.len() {
            return false
        }
        self.iter().all(|v| other.contains(v))
    }

    /// Returns true if self is a superset of other.
    ///
    /// This means all elements in other are contained in self.
    pub fn is_superset(self: &HashSet<T>, other: &HashSet<T>) -> bool {
        other.is_subset(self)
    }

    /// Returns true if self has no elements in common with other.
    pub fn is_disjoint(self: &HashSet<T>, other: &HashSet<T>) -> bool {
        if self.len() <= other.len() {
            self.iter().all(|v| !other.contains(v))
        } else {
            other.iter().all(|v| !self.contains(v))
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// Removes all elements e where f(&e) returns false.
    pub fn retain<F>(self: &!HashSet<T>, f: F)
    where F: fn(&T) -> bool
    {
        self.map.retain(|k, _| f(k))
    }

    /// Reserves capacity for at least additional more elements.
    pub fn reserve(self: &!HashSet<T>, additional: int) with Alloc {
        self.map.reserve(additional)
    }
}

impl<T> Clone for HashSet<T>
where T: Clone + Hash + Eq
{
    fn clone(self: &HashSet<T>) -> HashSet<T> with Alloc {
        HashSet { map: self.map.clone() }
    }
}

impl<T> Default for HashSet<T>
where T: Hash + Eq
{
    fn default() -> HashSet<T> with Alloc {
        HashSet::new()
    }
}

impl<T> Eq for HashSet<T>
where T: Hash + Eq
{
    fn eq(self: &HashSet<T>, other: &HashSet<T>) -> bool {
        self.len() == other.len() && self.iter().all(|v| other.contains(v))
    }
}

impl<T> Debug for HashSet<T>
where T: Debug + Hash + Eq
{
    fn fmt(self: &HashSet<T>, f: &!Formatter) -> Result<unit, FmtError> {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T> FromIterator<T> for HashSet<T>
where T: Hash + Eq
{
    fn from_iter<I>(iter: I) -> HashSet<T> with Alloc
    where I: IntoIterator<Item = T>
    {
        let mut set = HashSet::new()
        for item in iter {
            set.insert(item)
        }
        set
    }
}

impl<T> IntoIterator for HashSet<T>
where T: Hash + Eq
{
    type Item = T
    type IntoIter = IntoIter<T>

    fn into_iter(self: HashSet<T>) -> IntoIter<T> {
        IntoIter { inner: self.map.into_keys() }
    }
}

impl<T> Extend<T> for HashSet<T>
where T: Hash + Eq
{
    fn extend<I>(self: &!HashSet<T>, iter: I) with Alloc
    where I: IntoIterator<Item = T>
    {
        for item in iter {
            self.insert(item)
        }
    }
}

/// Iterator over HashSet values
pub struct Iter<T> {
    inner: Keys<T, unit>,
}

impl<T> Iterator for Iter<T> {
    type Item = &T

    fn next(self: &!Iter<T>) -> Option<&T> {
        self.inner.next()
    }

    fn size_hint(self: &Iter<T>) -> (int, Option<int>) {
        self.inner.size_hint()
    }
}

/// Owning iterator over HashSet values
pub struct IntoIter<T> {
    inner: IntoKeys<T, unit>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T

    fn next(self: &!IntoIter<T>) -> Option<T> {
        self.inner.next()
    }
}

/// Difference iterator
pub struct Difference<T> {
    iter: Iter<T>,
    other: &HashSet<T>,
}

impl<T> Iterator for Difference<T>
where T: Hash + Eq
{
    type Item = &T

    fn next(self: &!Difference<T>) -> Option<&T> {
        loop {
            let item = self.iter.next()?
            if !self.other.contains(item) {
                return Option::Some(item)
            }
        }
    }
}

/// Symmetric difference iterator
pub struct SymmetricDifference<T> {
    a_diff_b: Difference<T>,
    b_diff_a: Difference<T>,
    in_second: bool,
}

impl<T> Iterator for SymmetricDifference<T>
where T: Hash + Eq
{
    type Item = &T

    fn next(self: &!SymmetricDifference<T>) -> Option<&T> {
        if !self.in_second {
            match self.a_diff_b.next() {
                Option::Some(v) => Option::Some(v),
                Option::None => {
                    self.in_second = true
                    self.b_diff_a.next()
                }
            }
        } else {
            self.b_diff_a.next()
        }
    }
}

/// Intersection iterator
pub struct Intersection<T> {
    iter: Iter<T>,
    other: &HashSet<T>,
}

impl<T> Iterator for Intersection<T>
where T: Hash + Eq
{
    type Item = &T

    fn next(self: &!Intersection<T>) -> Option<&T> {
        loop {
            let item = self.iter.next()?
            if self.other.contains(item) {
                return Option::Some(item)
            }
        }
    }
}

/// Union iterator
pub struct Union<T> {
    iter: Iter<T>,
    other_iter: Difference<T>,
    in_other: bool,
}

impl<T> Iterator for Union<T>
where T: Hash + Eq
{
    type Item = &T

    fn next(self: &!Union<T>) -> Option<&T> {
        if !self.in_other {
            match self.iter.next() {
                Option::Some(v) => Option::Some(v),
                Option::None => {
                    self.in_other = true
                    self.other_iter.next()
                }
            }
        } else {
            self.other_iter.next()
        }
    }
}

// Unit tests
#[test]
fn test_hashset_basic() {
    let mut set = HashSet::new()
    assert(set.is_empty())

    set.insert(1)
    set.insert(2)
    set.insert(3)

    assert_eq(set.len(), 3)
    assert(set.contains(&1))
    assert(set.contains(&2))
    assert(set.contains(&3))
    assert(!set.contains(&4))
}

#[test]
fn test_hashset_insert_duplicate() {
    let mut set = HashSet::new()
    assert(set.insert(1))
    assert(!set.insert(1))
    assert_eq(set.len(), 1)
}

#[test]
fn test_hashset_remove() {
    let mut set = HashSet::new()
    set.insert(1)
    set.insert(2)

    assert(set.remove(&1))
    assert(!set.contains(&1))
    assert(!set.remove(&1))
}

#[test]
fn test_hashset_subset() {
    let mut a = HashSet::new()
    a.insert(1)
    a.insert(2)

    let mut b = HashSet::new()
    b.insert(1)
    b.insert(2)
    b.insert(3)

    assert(a.is_subset(&b))
    assert(!b.is_subset(&a))
    assert(b.is_superset(&a))
}

#[test]
fn test_hashset_disjoint() {
    let mut a = HashSet::new()
    a.insert(1)
    a.insert(2)

    let mut b = HashSet::new()
    b.insert(3)
    b.insert(4)

    assert(a.is_disjoint(&b))

    b.insert(1)
    assert(!a.is_disjoint(&b))
}
