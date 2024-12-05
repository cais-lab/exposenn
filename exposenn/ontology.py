import abc

import owlready2


class BaseOntologyAnalyzer:
    """Abstract base class for ontology analysis.
    
    Sets the interface for ontology analyzers.

    Methods
    -------
    get_relevant(concept: str) -> list
        Abstract method that must be implemented in subclasses. It returns a list of concepts relevant to the
        input concept according to the ontology.
    get_relevant_statements(concepts: list) -> list
        Abstract method that must be implemented in subclasses. It returns a list of propositional logic statements
        involving the given list of concepts.
    """
    
    @abc.abstractmethod
    def get_relevant(self, concept: str) -> list:
        """Returns a list of relevant concepts."""
        pass
        
    @abc.abstractmethod
    def get_relevant_statements(self, concepts: list) -> list:
        """Returns a list of propositional logics constructs."""
        pass


class MockOntologyAnalyzer(BaseOntologyAnalyzer):
    """
    The implementation of `BaseOntologyAnalyzer` that performs ontology analysis using the `owlready2` library.

    This class is designed to load an ontology from a file, analyze the relationships between concepts, and provide
    relevant concepts and logical statements based on the ontology structure.

    Parameters
    ----------
    ontology_file : str
        The path to the ontology file (in OWL format) that should be loaded and analyzed.

    Attributes
    ----------
    ontology_file : str
        Path to the ontology file.
    ontology : owlready2.Ontology
        The ontology object loaded from the provided file, which can be queried for concepts and relationships.

    Methods
    -------
    get_concept(concept: str)
        Returns the ontology concept corresponding to the input string.
    get_relevant(concept: str) -> list
        Returns a list of relevant concepts related to the input concept based on ontology relationships.
    get_relevant_statements(concepts: list) -> list
        Returns a list of propositional logic statements involving the provided list of concepts.
    """

    def __init__(self, ontology_file):
        super().__init__()
        self.ontology_file = ontology_file
        self.ontology = owlready2.get_ontology(ontology_file).load()
        owlready2.sync_reasoner([self.ontology])
    
    def get_concept(self, concept: str):
        """
        Returns the ontology concept corresponding to the given concept string.

        Parameters
        ----------
        concept : str
            The name of the concept to retrieve from the ontology.

        Returns
        -------
        owlready2.entity
            The ontology class (concept) corresponding to the input name.

        Raises
        ------
        ValueError
            If the concept does not exist in the ontology.
        """

        con = self.ontology[concept]
        if con is None:
            raise ValueError(f'"{concept}": no such concept in the ontology.')
        return con
    
    def get_relevant(self, concept: str) -> list:
        """
        Returns a list of relevant concepts for the given concept based on `is_a` relationships in the ontology.

        Parameters
        ----------
        concept : str
            The name of the input concept for which to retrieve relevant concepts.

        Returns
        -------
        list
            A list of ontology concepts that are relevant to the input concept. This includes both direct
            superclasses (concepts in the `is_a` relationship) and other concepts that share common relationships.
        """

        con = self.get_concept(concept)
        
        # concept -> X
        one = con.is_a
        # Y -> X
        two = []
        for c in self.ontology.classes():
            if c == con:
                continue
            for cisa in c.is_a:
                if cisa in one and c not in two:
                    two.append(c)
        return one + two
    
    def get_relevant_statements(self, concepts: list) -> list:
        """
        Returns a list of logical statements (implications) for the provided concepts based on ontology relationships.

        Parameters
        ----------
        concepts : list
            A list of concepts for which to generate logical statements.

        Returns
        -------
        list
            A list of propositional logic statements, represented as tuples (operator, concept1, concept2).
            The operator is typically '->' representing an implication (concept1 implies concept2).
        """

        statements = []
        for c in self.ontology.classes():
            for cisa in c.is_a:
                if c in concepts and cisa in concepts:
                    statements.append(('->', c, cisa))
        return statements
