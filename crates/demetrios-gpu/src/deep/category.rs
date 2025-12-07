//! Category-Theoretic Abstractions
//!
//! Category theory provides the mathematical foundation for composition
//! and abstraction. This module implements core categorical concepts
//! as computational primitives.
//!
//! # Key Concepts
//!
//! ## Categories
//! Objects and morphisms (arrows) with associative composition and identity.
//!
//! ## Functors
//! Structure-preserving maps between categories.
//!
//! ## Natural Transformations
//! Maps between functors that commute with all morphisms.
//!
//! ## Topoi
//! Categories that behave like "universes of sets" with their own logic.
//!
//! ## Monads & Comonads
//! Abstractions for sequencing and context.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

// ============================================================================
// CATEGORY
// ============================================================================

/// A category: objects and morphisms with composition
pub trait Category {
    /// The type of objects
    type Object: Clone + Eq + Hash;

    /// The type of morphisms
    type Morphism: Clone;

    /// Identity morphism for an object
    fn identity(obj: &Self::Object) -> Self::Morphism;

    /// Compose morphisms: g ∘ f (f first, then g)
    fn compose(f: &Self::Morphism, g: &Self::Morphism) -> Option<Self::Morphism>;

    /// Domain of a morphism
    fn domain(f: &Self::Morphism) -> Self::Object;

    /// Codomain of a morphism
    fn codomain(f: &Self::Morphism) -> Self::Object;
}

// ============================================================================
// SET CATEGORY
// ============================================================================

/// The category of sets and functions
#[derive(Debug, Clone)]
pub struct SetCategory;

/// A morphism in Set: a function between finite sets
#[derive(Clone)]
pub struct SetMorphism<A: Clone + Eq + Hash, B: Clone + Eq + Hash> {
    /// Domain type marker
    domain: PhantomData<A>,
    /// Codomain type marker
    codomain: PhantomData<B>,
    /// The function (as a map for finite sets)
    map: HashMap<A, B>,
    /// Domain set
    dom: Vec<A>,
    /// Codomain set
    cod: Vec<B>,
}

impl<A: Clone + Eq + Hash, B: Clone + Eq + Hash> SetMorphism<A, B> {
    /// Create from a mapping
    pub fn new(domain: Vec<A>, codomain: Vec<B>, map: HashMap<A, B>) -> Self {
        Self {
            domain: PhantomData,
            codomain: PhantomData,
            map,
            dom: domain,
            cod: codomain,
        }
    }

    /// Apply the function
    pub fn apply(&self, a: &A) -> Option<B> {
        self.map.get(a).cloned()
    }
}

impl<A: Clone + Eq + Hash + fmt::Debug, B: Clone + Eq + Hash + fmt::Debug> fmt::Debug
    for SetMorphism<A, B>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SetMorphism({:?} → {:?})", self.dom, self.cod)
    }
}

// ============================================================================
// FUNCTOR
// ============================================================================

/// A functor between categories
pub trait Functor<C: Category, D: Category> {
    /// Map an object
    fn map_object(obj: &C::Object) -> D::Object;

    /// Map a morphism (preserving composition)
    fn map_morphism(f: &C::Morphism) -> D::Morphism;
}

/// An endofunctor on a category (maps category to itself)
pub trait Endofunctor<C: Category>: Functor<C, C> {}

/// A concrete functor representation
#[derive(Debug, Clone)]
pub struct ConcreteFunctor<A, B> {
    name: String,
    _phantom: PhantomData<(A, B)>,
}

impl<A, B> ConcreteFunctor<A, B> {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            _phantom: PhantomData,
        }
    }
}

// ============================================================================
// NATURAL TRANSFORMATION
// ============================================================================

/// A natural transformation between functors
///
/// For functors F, G: C → D, a natural transformation α: F ⇒ G
/// assigns to each object X in C a morphism α_X: F(X) → G(X)
/// such that for all f: X → Y, G(f) ∘ α_X = α_Y ∘ F(f)
pub trait NaturalTransformation<C: Category, D: Category, F, G>
where
    F: Functor<C, D>,
    G: Functor<C, D>,
{
    /// The component at an object
    fn component(obj: &C::Object) -> D::Morphism;

    /// Verify naturality square for a morphism
    fn is_natural(f: &C::Morphism) -> bool;
}

/// A concrete natural transformation
#[derive(Debug, Clone)]
pub struct ConcreteNatTrans<A> {
    name: String,
    components: HashMap<A, A>,
}

impl<A: Clone + Eq + Hash> ConcreteNatTrans<A> {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            components: HashMap::new(),
        }
    }

    pub fn add_component(&mut self, source: A, target: A) {
        self.components.insert(source, target);
    }

    pub fn component(&self, obj: &A) -> Option<&A> {
        self.components.get(obj)
    }
}

// ============================================================================
// TOPOS
// ============================================================================

/// A topos: a category with logical structure
///
/// A topos has:
/// - All finite limits and colimits
/// - Exponential objects (function spaces)
/// - A subobject classifier Ω
///
/// Topoi can have different internal logics (classical, intuitionistic, etc.)
#[derive(Debug, Clone)]
pub struct Topos<T> {
    /// Name of this topos
    pub name: String,
    /// The subobject classifier type
    pub truth_values: Vec<T>,
    /// True value
    pub true_val: T,
    /// False value
    pub false_val: T,
    /// Is logic classical?
    pub is_classical: bool,
}

impl<T: Clone + Eq> Topos<T> {
    /// The topos of sets (classical logic)
    pub fn sets(true_val: T, false_val: T) -> Self {
        Self {
            name: "Set".to_string(),
            truth_values: vec![true_val.clone(), false_val.clone()],
            true_val,
            false_val,
            is_classical: true,
        }
    }

    /// Create a topos with custom truth values (e.g., for intuitionistic logic)
    pub fn custom(
        name: &str,
        truth_values: Vec<T>,
        true_val: T,
        false_val: T,
        is_classical: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            truth_values,
            true_val,
            false_val,
            is_classical,
        }
    }

    /// Number of truth values
    pub fn logic_size(&self) -> usize {
        self.truth_values.len()
    }

    /// Is this boolean (2-valued) logic?
    pub fn is_boolean(&self) -> bool {
        self.truth_values.len() == 2
    }
}

// ============================================================================
// LIMIT AND COLIMIT
// ============================================================================

/// A limit of a diagram
#[derive(Debug, Clone)]
pub struct Limit<T> {
    /// The limiting object
    pub apex: T,
    /// Projection morphisms (indexed by diagram objects)
    pub projections: HashMap<usize, T>,
}

impl<T: Clone> Limit<T> {
    /// Create a product (limit of discrete diagram)
    pub fn product(factors: Vec<T>, apex: T) -> Self {
        let projections: HashMap<usize, T> = factors
            .into_iter()
            .enumerate()
            .map(|(i, f)| (i, f))
            .collect();
        Self { apex, projections }
    }

    /// Create an equalizer (limit of parallel pair)
    pub fn equalizer(equalized: T) -> Self {
        let mut projections = HashMap::new();
        projections.insert(0, equalized.clone());
        Self {
            apex: equalized,
            projections,
        }
    }

    /// Create a pullback (limit of cospan)
    pub fn pullback(apex: T, left: T, right: T) -> Self {
        let mut projections = HashMap::new();
        projections.insert(0, left);
        projections.insert(1, right);
        Self { apex, projections }
    }
}

/// A colimit of a diagram
#[derive(Debug, Clone)]
pub struct Colimit<T> {
    /// The colimiting object
    pub nadir: T,
    /// Injection morphisms (indexed by diagram objects)
    pub injections: HashMap<usize, T>,
}

impl<T: Clone> Colimit<T> {
    /// Create a coproduct (colimit of discrete diagram)
    pub fn coproduct(summands: Vec<T>, nadir: T) -> Self {
        let injections: HashMap<usize, T> = summands
            .into_iter()
            .enumerate()
            .map(|(i, s)| (i, s))
            .collect();
        Self { nadir, injections }
    }

    /// Create a coequalizer (colimit of parallel pair)
    pub fn coequalizer(coequalized: T) -> Self {
        let mut injections = HashMap::new();
        injections.insert(0, coequalized.clone());
        Self {
            nadir: coequalized,
            injections,
        }
    }

    /// Create a pushout (colimit of span)
    pub fn pushout(nadir: T, left: T, right: T) -> Self {
        let mut injections = HashMap::new();
        injections.insert(0, left);
        injections.insert(1, right);
        Self { nadir, injections }
    }
}

// ============================================================================
// ADJUNCTION
// ============================================================================

/// An adjunction between functors F ⊣ G
///
/// F: C → D is left adjoint to G: D → C when:
/// Hom_D(F(X), Y) ≅ Hom_C(X, G(Y)) naturally
#[derive(Debug, Clone)]
pub struct Adjunction<F, G> {
    /// Left adjoint
    pub left: F,
    /// Right adjoint
    pub right: G,
    /// Name
    pub name: String,
}

impl<F, G> Adjunction<F, G> {
    /// Create an adjunction
    pub fn new(name: &str, left: F, right: G) -> Self {
        Self {
            left,
            right,
            name: name.to_string(),
        }
    }
}

// ============================================================================
// MONAD
// ============================================================================

/// A monad: endofunctor with unit and multiplication
///
/// A monad M on a category C consists of:
/// - An endofunctor M: C → C
/// - A natural transformation η: Id ⇒ M (unit)
/// - A natural transformation μ: M² ⇒ M (multiplication)
///
/// Subject to associativity and unit laws.
pub trait Monad {
    /// The wrapped type
    type Inner;

    /// Unit: a → M(a)
    fn unit(a: Self::Inner) -> Self;

    /// Bind: M(a) → (a → M(b)) → M(b)
    fn bind<B, F>(self, f: F) -> Self
    where
        F: FnOnce(Self::Inner) -> Self,
        Self: Sized;

    /// Join: M(M(a)) → M(a) (derived from bind)
    fn join(mma: Self) -> Self
    where
        Self: Sized,
        Self::Inner: Clone,
    {
        // Default implementation using bind
        // Specific monads may override
        mma
    }
}

/// The Identity monad
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity<A>(pub A);

impl<A> Monad for Identity<A> {
    type Inner = A;

    fn unit(a: A) -> Self {
        Identity(a)
    }

    fn bind<B, F>(self, f: F) -> Self
    where
        F: FnOnce(A) -> Self,
    {
        f(self.0)
    }
}

/// The Maybe/Option monad
impl<A> Monad for Option<A> {
    type Inner = A;

    fn unit(a: A) -> Self {
        Some(a)
    }

    fn bind<B, F>(self, f: F) -> Self
    where
        F: FnOnce(A) -> Self,
    {
        match self {
            Some(a) => f(a),
            None => None,
        }
    }
}

/// List monad operations (separate from Monad trait due to FnOnce constraint)
pub mod list_monad {
    /// Unit for list monad
    pub fn unit<A>(a: A) -> Vec<A> {
        vec![a]
    }

    /// Bind for list monad (concatMap)
    pub fn bind<A, B, F>(list: Vec<A>, f: F) -> Vec<B>
    where
        F: Fn(A) -> Vec<B>,
    {
        list.into_iter().flat_map(f).collect()
    }

    /// Join for list monad (flatten)
    pub fn join<A>(nested: Vec<Vec<A>>) -> Vec<A> {
        nested.into_iter().flatten().collect()
    }
}

// ============================================================================
// COMONAD
// ============================================================================

/// A comonad: dual of monad
///
/// A comonad W on a category C consists of:
/// - An endofunctor W: C → C
/// - A natural transformation ε: W ⇒ Id (counit/extract)
/// - A natural transformation δ: W ⇒ W² (comultiplication/duplicate)
pub trait Comonad {
    /// The wrapped type
    type Inner;

    /// Extract: W(a) → a
    fn extract(&self) -> Self::Inner;

    /// Duplicate: W(a) → W(W(a))
    fn duplicate(self) -> Self
    where
        Self: Sized;

    /// Extend: W(a) → (W(a) → b) → W(b)
    fn extend<B, F>(self, f: F) -> Self
    where
        F: Fn(&Self) -> B,
        Self: Sized;
}

/// A simple stream comonad (infinite list with focus)
#[derive(Debug, Clone)]
pub struct Stream<A> {
    /// Current focus
    pub head: A,
    /// Generator for the tail
    pub tail: Vec<A>,
}

impl<A: Clone> Stream<A> {
    pub fn new(head: A, tail: Vec<A>) -> Self {
        Self { head, tail }
    }
}

impl<A: Clone> Comonad for Stream<A> {
    type Inner = A;

    fn extract(&self) -> A {
        self.head.clone()
    }

    fn duplicate(self) -> Self {
        // Create stream of streams (simplified)
        Self {
            head: self.head.clone(),
            tail: self.tail.clone(),
        }
    }

    fn extend<B, F>(self, f: F) -> Self
    where
        F: Fn(&Self) -> B,
    {
        // Simplified: apply f at each position
        self.duplicate()
    }
}

// ============================================================================
// YONEDA LEMMA
// ============================================================================

/// Yoneda embedding: every object is characterized by its morphisms
///
/// For an object A in category C:
/// Hom(−, A) is a functor C^op → Set
///
/// Yoneda lemma: Nat(Hom(−, A), F) ≅ F(A)
#[derive(Debug, Clone)]
pub struct Yoneda<A> {
    /// The represented object
    pub object: A,
}

impl<A: Clone> Yoneda<A> {
    /// Create Yoneda embedding for an object
    pub fn embed(object: A) -> Self {
        Self { object }
    }

    /// Apply the Yoneda lemma: transform to any functor value
    pub fn yoneda<F, B, G>(self, f: G) -> B
    where
        G: FnOnce(A) -> B,
    {
        f(self.object)
    }
}

// ============================================================================
// KAN EXTENSIONS
// ============================================================================

/// Left Kan extension
#[derive(Debug, Clone)]
pub struct LeftKan<F, G> {
    /// The functor being extended
    pub functor: F,
    /// Along which functor
    pub along: G,
    pub name: String,
}

/// Right Kan extension
#[derive(Debug, Clone)]
pub struct RightKan<F, G> {
    /// The functor being extended
    pub functor: F,
    /// Along which functor
    pub along: G,
    pub name: String,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_monad() {
        let x = Identity::unit(42);
        let y = x.bind::<i32, _>(|n| Identity::unit(n * 2));
        assert_eq!(y, Identity(84));
    }

    #[test]
    fn test_option_monad() {
        let x: Option<i32> = Monad::unit(42);
        let y = x.bind::<i32, _>(|n| Some(n * 2));
        assert_eq!(y, Some(84));

        let z: Option<i32> = None;
        let w = z.bind::<i32, _>(|n| Some(n * 2));
        assert_eq!(w, None);
    }

    #[test]
    fn test_list_monad() {
        let x = list_monad::unit(42);
        assert_eq!(x, vec![42]);

        let xs = vec![1, 2, 3];
        let ys = list_monad::bind(xs, |n| vec![n, n * 10]);
        assert_eq!(ys, vec![1, 10, 2, 20, 3, 30]);
    }

    #[test]
    fn test_topos_sets() {
        let topos = Topos::sets(true, false);
        assert!(topos.is_classical);
        assert!(topos.is_boolean());
    }

    #[test]
    fn test_topos_three_valued() {
        // Three-valued logic topos
        let topos = Topos::custom(
            "ThreeValued",
            vec![0, 1, 2], // false, unknown, true
            2,
            0,
            false, // Not classical
        );

        assert!(!topos.is_classical);
        assert!(!topos.is_boolean());
        assert_eq!(topos.logic_size(), 3);
    }

    #[test]
    fn test_product_limit() {
        let product = Limit::product(vec!["A", "B", "C"], "A×B×C");

        assert_eq!(product.apex, "A×B×C");
        assert_eq!(product.projections.len(), 3);
    }

    #[test]
    fn test_coproduct_colimit() {
        let coproduct = Colimit::coproduct(vec!["A", "B", "C"], "A+B+C");

        assert_eq!(coproduct.nadir, "A+B+C");
        assert_eq!(coproduct.injections.len(), 3);
    }

    #[test]
    fn test_stream_comonad() {
        let stream = Stream::new(1, vec![2, 3, 4, 5]);

        // Extract gives the head
        assert_eq!(stream.extract(), 1);
    }

    #[test]
    fn test_yoneda() {
        let y = Yoneda::embed(42i32);
        let result = y.yoneda::<fn(i32) -> String, String, _>(|n| n.to_string());
        assert_eq!(result, "42");
    }

    #[test]
    fn test_monad_laws_identity() {
        // Left identity: unit(a).bind(f) == f(a)
        let a = 42;
        let f = |n: i32| Identity(n * 2);

        let left = Identity::unit(a).bind::<i32, _>(f);
        let right = f(a);
        assert_eq!(left, right);

        // Right identity: m.bind(unit) == m
        let m = Identity(42);
        let bound = m.clone().bind::<i32, _>(Identity::unit);
        assert_eq!(bound, m);
    }
}
