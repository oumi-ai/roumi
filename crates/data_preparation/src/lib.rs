use tch::Tensor; 
use std::collections::HashMap;

// Represents a dataset as a dictionary of lists of tensors 
#[derive(Debug)]
pub struct Dataset{
    pub tensors: HashMap<String, Vec<Tensor>>,
}

impl Dataset{
    pub fn new(tensors: HashMap<String, Vec<Tensor>>) -> Self{
        Dataset{tensors}
    }

    pub fn len(&self) -> usize {
        self.tensors.values().next().map_or(0, |v| v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}


