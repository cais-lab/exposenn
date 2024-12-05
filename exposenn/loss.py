import os
import subprocess
import torch
import semantic_loss_pytorch
import tempfile
from semantic_loss_pytorch.constraints_to_cnf import ConstraintsToCnf


class SemanticLoss:
    """
    A class that implements Semantic Loss, which enforces logical constraints on the output predictions of a neural network.
    This class uses a logical theory defined over output concepts and converts it into a form that can be used as a
    differentiable loss function, ensuring that the model's predictions adhere to specific logical rules.

    The class converts the logical theory into a Conjunctive Normal Form (CNF), then further processes it into a
    Sentential Decision Diagram (SDD) to compute the semantic loss efficiently.

    Parameters
    ----------
    output_concepts : list
        A list of output concepts corresponding to the predictions of the model.
    logical_theory : list[tuple]
        A list of tuples representing the logical theory that defines relationships between output concepts. Each tuple
        should represent a logical rule in the format (operator, concept1, concept2), where `operator` is a logical
        operator (e.g., '->' for implication), and `concept1` and `concept2` are concepts from the output concept list.
    target_concept : optional
        A target concept predicted by the base network that can be taken into account in the semantic loss calculation if given.

    Attributes
    ----------
    target_concept : str or None
        The name of the target concept if provided, otherwise `None`.
    loss_fn : semantic_loss_pytorch.SemanticLoss
        The SemanticLoss function that is initialized after converting the logical theory to CNF and SDD. This function
        computes the actual semantic loss for model predictions.

    Methods
    -------
    __call__(prediction, ignored)
        Computes the semantic loss for the given model predictions. If a target concept is defined, the base network predictions
        are concatenated with the interpretation network predictions before applying the loss function.

    Example
    -------
    output_concepts = ['Concept1', 'Concept2', 'Concept3']
    logical_theory = [('->', 'Concept1', 'Concept2')]  # Concept1 implies Concept2
    semantic_loss = SemanticLoss(output_concepts, logical_theory)

    Notes
    -----
    - The logical theory must be expressed as tuples where the first element is the operator (currently only '->' is
      supported), and the second and third elements are the concepts involved in the relationship.
    - The class uses external tools to convert logical constraints into CNF format and then to SDD format.

    Raises
    ------
    ValueError
        If an unsupported logical operator is encountered in the logical theory.
    """

    def __init__(self, output_concepts: list, logical_theory: list[tuple], target_concept=None):

        self.target_concept = None

        if target_concept is not None:
            output_concepts.append(target_concept)
            self.target_concept = target_concept

        var_dict = {concept: f'X{i}' for i, concept in enumerate(output_concepts)}
        rel_statements = [t for t in logical_theory if t[1] in output_concepts and t[2] in output_concepts]

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_constraints_path = os.path.join(temp_dir, 'raw_constraints.txt')
            dimacs_cnf_path = os.path.join(temp_dir, 'dimacs_cnf.txt')
            vtree_path = os.path.join(temp_dir, 'constraint.vtree')
            sdd_path = os.path.join(temp_dir, 'constraint.sdd')

            with open(raw_constraints_path, "w") as file:
                file.write(f'shape [{len(output_concepts)}]' + '\n')

                for statement in rel_statements:
                    operator = statement[0]
                    if operator == '->':
                        file.write('\n' + var_dict[statement[1]] + ' >> ' + var_dict[statement[2]])
                    else:
                        raise ValueError(f"Unsupported operator: {operator}")

            ConstraintsToCnf.expression_to_cnf(raw_constraints_path, dimacs_cnf_path, 1)

            build_trees = ['pysdd', '-c', dimacs_cnf_path, '-W', vtree_path, '-R', sdd_path]
            subprocess.run(build_trees, capture_output=True, text=True)

            self.loss_fn = semantic_loss_pytorch.SemanticLoss(sdd_path, vtree_path)

    def __call__(self, prediction, ignored):
        if self.target_concept is not None:
            prediction = torch.cat((prediction, ignored), dim=1)

        return self.loss_fn(probabilities=prediction.T)


class AdditiveMultipartLoss:
    """
    A composite loss function that combines two separate loss functions
    (one for concepts and one for targets) in a weighted additive manner.

    Parameters
    ----------
    concepts_loss_fn : Callable
        The loss function for concepts. This function should accept two arguments:
        the predicted concepts and the ground truth concepts.
    target_loss_fn : Callable
        The loss function for the target. This function should accept two arguments:
        the predicted target and the ground truth target.
    concepts_coef : float, optional
        Weight coefficient for the concepts loss term. Defaults to 1.0.
    target_coef : float, optional
        Weight coefficient for the target loss term. Defaults to 1.0.

    Methods
    -------
    __call__(concepts, target)
        Computes the combined loss as a weighted sum of the concepts loss and the target loss.
    """

    def __init__(self, *, concepts_loss_fn,
                 target_loss_fn,
                 concepts_coef=1.0,
                 target_coef=1.0):
        self.concepts_loss_fn = concepts_loss_fn
        self.target_loss_fn = target_loss_fn
        self.concepts_coef = concepts_coef
        self.target_coef = target_coef

    def __call__(self, concepts, target):
        return (self.target_coef * self.target_loss_fn(*target) +
                self.concepts_coef * self.concepts_loss_fn(*concepts))