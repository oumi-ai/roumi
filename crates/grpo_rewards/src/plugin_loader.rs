use crate::rewards::Calculator;
use anyhow::Result;
use libloading::{Library, Symbol};
use std::collections::HashMap;

// Define a struct to hold loaded libraries
pub struct PluginLoader {
    // Store libraries to keep them loaded
    _libraries: Vec<Library>,
    calculator_creators:
        HashMap<String, Symbol<'static, fn(HashMap<String, String>) -> Box<dyn Calculator>>>,
}

impl PluginLoader {
    pub fn new() -> Self {
        PluginLoader {
            _libraries: Vec::new(),
            calculator_creators: HashMap::new(),
        }
    }

    /// Loads a calculator plugin from the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the dynamic library (.so, .dll, .dylib) containing the plugin
    ///
    /// # Returns
    ///
    /// A Result containing a vector of calculator names that were successfully loaded
    /// from the plugin, or an error if the plugin could not be loaded.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it loads and executes code from a dynamic library.
    /// The caller must ensure that the library at the given path is a valid calculator plugin
    /// that correctly implements the expected interface.
    pub fn load_plugin(&mut self, path: &str) -> Result<Vec<String>> {
        unsafe {
            // Load the dynamic library
            let lib = Library::new(path)?;

            // Get the registration function
            let register_fn: Symbol<
                fn() -> Vec<(
                    &'static str,
                    fn(HashMap<String, String>) -> Box<dyn Calculator>,
                )>,
            > = lib.get(b"register_calculators")?;

            // Call the registration function to get calculator creators
            let calculators = register_fn();
            let mut calculator_names = Vec::new();

            // Store each calculator creator
            for (name, creator) in calculators {
                self.calculator_creators
                    .insert(name.to_string(), std::mem::transmute(creator));
                calculator_names.push(name.to_string());
            }

            // Keep the library loaded
            self._libraries.push(lib);

            Ok(calculator_names)
        }
    }

    /// Creates a calculator instance by name with the provided parameters.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the calculator to create, as returned by `load_plugin`
    /// * `params` - A hashmap of string parameters to configure the calculator
    ///
    /// # Returns
    ///
    /// Some(Box<dyn Calculator>) if the calculator was found and successfully created,
    /// or None if no calculator with the specified name exists.
    pub fn create_calculator(
        &self,
        name: &str,
        params: HashMap<String, String>,
    ) -> Option<Box<dyn Calculator>> {
        self.calculator_creators
            .get(name)
            .map(|creator| creator(params))
    }
}
