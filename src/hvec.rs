use core::{
    cmp,
    fmt::Debug,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
};
use header_vec::HeaderVecWeak;

#[derive(Debug)]
pub(crate) struct HggEdge<K> {
    pub(crate) key: K,
    pub(crate) neighbor: HVec<K>,
}

#[derive(Debug)]
pub(crate) struct HggHeader<K> {
    pub(crate) key: K,
    pub(crate) node: usize,
}

#[derive(Debug)]
pub(crate) struct HVec<K>(pub(crate) HeaderVecWeak<HggHeader<K>, HggEdge<K>>);

impl<K> HVec<K> {
    pub fn weak(&self) -> Self {
        unsafe { Self(self.0.weak()) }
    }

    pub fn contains(&self, other: &Self) -> bool {
        self.as_slice()
            .iter()
            .any(|edge| edge.neighbor.is(other.ptr()))
    }
}

impl<K> Deref for HVec<K> {
    type Target = HeaderVecWeak<HggHeader<K>, HggEdge<K>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K> DerefMut for HVec<K> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<K> PartialEq for HVec<K> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr().eq(&other.ptr())
    }
}

impl<K> Eq for HVec<K> {}

impl<K> PartialOrd for HVec<K> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.ptr().partial_cmp(&other.ptr())
    }

    fn lt(&self, other: &Self) -> bool {
        self.ptr().lt(&other.ptr())
    }
    fn le(&self, other: &Self) -> bool {
        self.ptr().le(&other.ptr())
    }
    fn gt(&self, other: &Self) -> bool {
        self.ptr().gt(&other.ptr())
    }
    fn ge(&self, other: &Self) -> bool {
        self.ptr().ge(&other.ptr())
    }
}

impl<K> Ord for HVec<K> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.ptr().cmp(&other.ptr())
    }
}

impl<K> Hash for HVec<K> {
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        self.ptr().hash(hasher);
    }
}
