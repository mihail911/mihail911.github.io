import lodash from 'lodash';
import path from "path"

/**
 * Finds the appropriate tag slug for the given tag from a list of all tag
 * slugs for a given post.
 * @param tag Tag to find slug for
 * @param tagSlugs List of all tag slugs for given post
 * @returns {*}
 */
export function findMatchingTagSlug(tag, tagSlugs) {
  const formattedTag = lodash.kebabCase(tag);
  let matchingTagSlug = null;
  for (let tagSlugIdx = 0; tagSlugIdx < tagSlugs.length; tagSlugIdx++) {
    const tagSlug = tagSlugs[tagSlugIdx];
    if (path.basename(tagSlug) === formattedTag) {
      matchingTagSlug = tagSlug;
      return matchingTagSlug;
    }
  }
  return matchingTagSlug;
}

