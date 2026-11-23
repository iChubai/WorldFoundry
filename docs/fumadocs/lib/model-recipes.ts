import type { ModelRecipe, ModelRecipeData } from '@/lib/model-recipe-types';
import { readSiteJson } from '@/lib/read-site-json';

export const modelRecipeData = readSiteJson<ModelRecipeData>('lib/model-recipes-data.json');

const recipeById = new Map(modelRecipeData.recipes.map((recipe) => [recipe.id, recipe]));

export function getModelRecipe(modelId: string) {
  return recipeById.get(modelId);
}

const GENERIC_TASKS = new Set(['world', 'video', 'multimodal', 'three_dimension']);

function familyPrefix(id: string) {
  return id.split(/[-_.]/)[0]?.replace(/\d+$/, '') ?? id;
}

export function getRelatedModelRecipes(recipe: ModelRecipe, limit = 4) {
  const prefix = familyPrefix(recipe.id);
  const taskSet = new Set(
    recipe.tasks.map((task) => task.toLowerCase()).filter((task) => !GENERIC_TASKS.has(task)),
  );

  return modelRecipeData.recipes
    .filter((candidate) => candidate.id !== recipe.id && candidate.category === recipe.category)
    .map((candidate) => {
      let score = 0;
      if (prefix.length >= 3 && familyPrefix(candidate.id) === prefix) score += 10;
      if (recipe.provider && candidate.provider === recipe.provider) score += 6;
      for (const task of candidate.tasks) {
        if (taskSet.has(task.toLowerCase())) score += 2;
      }
      if (candidate.status.group === 'verified') score += 2;
      else if (candidate.status.group === 'integrated') score += 1;
      return { candidate, score };
    })
    .sort((left, right) => right.score - left.score || left.candidate.name.localeCompare(right.candidate.name))
    .slice(0, limit)
    .map((entry) => entry.candidate);
}
