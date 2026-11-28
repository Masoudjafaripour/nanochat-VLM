//! Bit-exact port of the parts of CPython's `random.Random` that CORE eval uses:
//! seeding from an int, `shuffle`, and `sample`. Needed so the Rust eval picks the
//! same data order and the same few-shot examples as `core_eval.py` / `base_eval.py`.

pub struct PyRandom {
    mt: [u32; 624],
    idx: usize,
}

impl PyRandom {
    /// `random.Random(seed)` for a non-negative int seed (init_by_array on its 32-bit words).
    pub fn new(seed: u64) -> Self {
        let key: Vec<u32> = if seed >> 32 == 0 { vec![seed as u32] } else { vec![seed as u32, (seed >> 32) as u32] };
        let mut r = Self { mt: [0; 624], idx: 624 };
        r.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..624.max(key.len()) {
            let prev = r.mt[i - 1] ^ (r.mt[i - 1] >> 30);
            r.mt[i] = (r.mt[i] ^ prev.wrapping_mul(1664525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let prev = r.mt[i - 1] ^ (r.mt[i - 1] >> 30);
            r.mt[i] = (r.mt[i] ^ prev.wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
        }
        r.mt[0] = 0x8000_0000;
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            self.mt[i] = 1812433253u32.wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        self.idx = 624;
    }

    fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[k] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `_randbelow_with_getrandbits(n)` (n < 2^32).
    fn randbelow(&mut self, n: usize) -> usize {
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = (self.next_u32() >> (32 - k)) as usize;
            if r < n {
                return r;
            }
        }
    }

    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i + 1);
            x.swap(i, j);
        }
    }

    /// `random.sample(range-like population of len n, k)` -> chosen indices.
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n);
        let mut setsize = 21usize;
        if k > 5 {
            setsize += 4usize.pow(((3 * k) as f64).log(4.0).ceil() as u32);
        }
        let mut out = Vec::with_capacity(k);
        if n <= setsize {
            let mut pool: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = self.randbelow(n - i);
                out.push(pool[j]);
                pool[j] = pool[n - i - 1];
            }
        } else {
            let mut selected = std::collections::HashSet::new();
            for _ in 0..k {
                let mut j = self.randbelow(n);
                while selected.contains(&j) {
                    j = self.randbelow(n);
                }
                selected.insert(j);
                out.push(j);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values printed by CPython's `random` module.
    #[test]
    fn matches_cpython() {
        let mut x: Vec<usize> = (0..10).collect();
        PyRandom::new(1337).shuffle(&mut x);
        assert_eq!(x, vec![0, 3, 7, 2, 1, 6, 4, 5, 8, 9]);
        assert_eq!(PyRandom::new(1234).sample_indices(10, 3), vec![7, 1, 0]); // pool branch
        assert_eq!(PyRandom::new(1239).sample_indices(200, 10), vec![19, 71, 116, 6, 12, 26, 122, 123, 175, 164]); // set branch
        assert_eq!(PyRandom::new((1u64 << 40) + 5).sample_indices(50, 6), vec![32, 33, 17, 42, 1, 48]); // 2-word seed
    }
}
